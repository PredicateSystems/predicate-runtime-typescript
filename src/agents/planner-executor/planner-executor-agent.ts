/**
 * PlannerExecutorAgent: Two-tier agent architecture for browser automation.
 *
 * MVP implementation with:
 * - Stepwise (ReAct-style) planning
 * - Compact context formatting for small models
 * - Snapshot escalation for reliable element capture
 * - Pre-step verification
 * - Retry/repair logic
 * - Token usage tracking
 * - Pre-action authorization hook (for sidecar policy integration)
 *
 * Deferred to post-MVP:
 * - Vision fallback
 * - Modal/overlay dismissal
 * - Captcha handling
 * - Intent heuristics
 * - Recovery navigation
 */

import type { LLMProvider, LLMResponse } from '../../llm-provider';
import type {
  PlannerExecutorConfig,
  SnapshotEscalationConfig,
  RetryConfig,
  StepwisePlanningConfig,
} from './config';
import { DEFAULT_CONFIG, mergeConfig, type DeepPartial } from './config';
import type {
  Plan,
  PlanStep,
  ActionRecord,
  StepOutcome,
  RunOutcome,
  TokenUsageSummary,
  TokenUsageTotals,
  SnapshotContext,
  ParsedAction,
  Snapshot,
  SnapshotElement,
} from './plan-models';
import { StepStatus, PlanSchema } from './plan-models';
import {
  buildStepwisePlannerPrompt,
  buildExecutorPrompt,
  type StepwisePlannerResponse,
} from './prompts';
import { parseAction, extractJson, normalizePlan, formatContext } from './plan-utils';

// ---------------------------------------------------------------------------
// Token Usage Collector
// ---------------------------------------------------------------------------

/**
 * Collects token usage statistics by role (planner/executor) and model.
 */
class TokenUsageCollector {
  private byRole: Map<string, TokenUsageTotals> = new Map();
  private byModel: Map<string, TokenUsageTotals> = new Map();

  record(role: string, resp: LLMResponse): void {
    // By role
    const roleTotals = this.byRole.get(role) || {
      calls: 0,
      promptTokens: 0,
      completionTokens: 0,
      totalTokens: 0,
    };
    roleTotals.calls += 1;
    const pt = typeof resp.promptTokens === 'number' ? Math.max(0, resp.promptTokens) : 0;
    const ct = typeof resp.completionTokens === 'number' ? Math.max(0, resp.completionTokens) : 0;
    const tt = typeof resp.totalTokens === 'number' ? Math.max(0, resp.totalTokens) : pt + ct;
    roleTotals.promptTokens += pt;
    roleTotals.completionTokens += ct;
    roleTotals.totalTokens += tt;
    this.byRole.set(role, roleTotals);

    // By model
    const modelName = (resp.modelName || '').trim() || 'unknown';
    const modelTotals = this.byModel.get(modelName) || {
      calls: 0,
      promptTokens: 0,
      completionTokens: 0,
      totalTokens: 0,
    };
    modelTotals.calls += 1;
    modelTotals.promptTokens += pt;
    modelTotals.completionTokens += ct;
    modelTotals.totalTokens += tt;
    this.byModel.set(modelName, modelTotals);
  }

  reset(): void {
    this.byRole.clear();
    this.byModel.clear();
  }

  summary(): TokenUsageSummary {
    // Sum totals
    const total: TokenUsageTotals = {
      calls: 0,
      promptTokens: 0,
      completionTokens: 0,
      totalTokens: 0,
    };
    for (const t of this.byRole.values()) {
      total.calls += t.calls;
      total.promptTokens += t.promptTokens;
      total.completionTokens += t.completionTokens;
      total.totalTokens += t.totalTokens;
    }

    // Convert maps to records
    const byRole: Record<string, TokenUsageTotals> = {};
    for (const [k, v] of this.byRole) {
      byRole[k] = { ...v };
    }
    const byModel: Record<string, TokenUsageTotals> = {};
    for (const [k, v] of this.byModel) {
      byModel[k] = { ...v };
    }

    return { total, byRole, byModel };
  }
}

// ---------------------------------------------------------------------------
// Pre-Action Authorizer Interface
// ---------------------------------------------------------------------------

/**
 * Authorization result from pre-action check.
 */
export interface AuthorizationResult {
  /** Whether the action is allowed */
  allowed: boolean;
  /** Reason for denial (if not allowed) */
  reason?: string;
  /** Alternative action to take (if any) */
  alternative?: string;
}

/**
 * Hook for pre-action authorization (e.g., sidecar policy evaluation).
 *
 * Called before each action is executed. If the action is denied,
 * the agent will skip the action and record the denial.
 *
 * @example
 * ```typescript
 * const authorizer: PreActionAuthorizer = async (action, context) => {
 *   const response = await fetch('http://localhost:3500/v1/authorize', {
 *     method: 'POST',
 *     body: JSON.stringify({
 *       principal: 'agent:browser-automation',
 *       action: `browser.${action.type.toLowerCase()}`,
 *       resource: context.url,
 *     }),
 *   });
 *   const result = await response.json();
 *   return { allowed: result.decision === 'ALLOW', reason: result.reason };
 * };
 * ```
 */
export type PreActionAuthorizer = (
  action: { type: string; elementId?: number; value?: string },
  context: { url: string; stepGoal: string; taskGoal: string }
) => Promise<AuthorizationResult>;

// ---------------------------------------------------------------------------
// Intent Heuristics Interface
// ---------------------------------------------------------------------------

/**
 * Interface for intent-based element selection.
 *
 * Allows bypassing LLM for known patterns (e.g., "add to cart" buttons).
 * Can be implemented by domain-specific heuristics.
 */
export interface IntentHeuristics {
  /**
   * Find element matching the given intent.
   *
   * @param intent - Intent string (e.g., "add_to_cart", "search")
   * @param elements - Available elements from snapshot
   * @param url - Current page URL
   * @param goal - Step goal description
   * @returns Element ID if found, null otherwise
   */
  findElementForIntent(
    intent: string,
    elements: SnapshotElement[],
    url: string,
    goal: string
  ): number | null;

  /**
   * Get priority order for intent matching.
   * Higher priority intents are tried first.
   */
  priorityOrder(): string[];
}

/**
 * Common text patterns for intent matching.
 * Used as fallback when no custom heuristics are provided.
 */
const COMMON_INTENT_PATTERNS: Record<string, string[]> = {
  add_to_cart: ['add to cart', 'add to bag', 'add to basket', 'buy now', 'add item'],
  checkout: ['checkout', 'proceed to checkout', 'go to checkout', 'check out'],
  search: ['search', 'find', 'go', 'submit'],
  login: ['log in', 'login', 'sign in', 'signin'],
  submit: ['submit', 'send', 'continue', 'next', 'confirm'],
  close: ['close', 'dismiss', 'x', 'cancel', 'no thanks'],
};

/**
 * Action verbs to strip from descriptive intents.
 * E.g., "click the Add to Cart button" → "add to cart button"
 */
const ACTION_VERBS = [
  'click',
  'tap',
  'press',
  'select',
  'choose',
  'pick',
  'find',
  'locate',
  'look for',
  'search for',
  'type',
  'enter',
  'input',
  'fill',
  'scroll to',
  'navigate to',
  'go to',
  'open',
  'close',
  'dismiss',
  'accept',
  'the',
  'a',
  'an',
  'on',
  'button',
  'link',
  'field',
  'input',
  'element',
];

/**
 * Extract meaningful keywords from a descriptive intent.
 * Handles cases like:
 * - "click the Add to Cart button" → "add to cart"
 * - 'click "Add to Cart"' → "add to cart"
 * - "Add to Cart" → "add to cart"
 */
function extractIntentKeywords(intent: string): string[] {
  let normalized = intent.toLowerCase().trim();

  // Extract quoted text first (e.g., 'click "Add to Cart"')
  const quotedMatch = normalized.match(/["']([^"']+)["']/);
  if (quotedMatch) {
    return [quotedMatch[1].trim()];
  }

  // Strip action verbs from the beginning
  for (const verb of ACTION_VERBS) {
    const pattern = new RegExp(`^${verb}\\s+`, 'i');
    normalized = normalized.replace(pattern, '');
  }

  // Strip trailing words like "button", "link", "element"
  normalized = normalized.replace(/\s+(button|link|element|field|input)$/i, '');

  // Clean up extra spaces
  normalized = normalized.replace(/\s+/g, ' ').trim();

  // If the result is too short, return the original
  if (normalized.length < 2) {
    return [intent.toLowerCase().replace(/[_-]/g, ' ')];
  }

  return [normalized];
}

/**
 * Simple intent heuristics using text pattern matching.
 * Used as default when no custom heuristics are provided.
 *
 * This is GENERALIZABLE - it works for any intent by:
 * 1. Checking common patterns (add_to_cart, checkout, etc.)
 * 2. Extracting keywords from descriptive intents (e.g., "click the X button")
 * 3. Falling back to direct text matching
 */
class SimpleIntentHeuristics implements IntentHeuristics {
  findElementForIntent(
    intent: string,
    elements: SnapshotElement[],
    _url: string,
    _goal: string
  ): number | null {
    const normalizedIntent = intent.toLowerCase().replace(/[_-]/g, ' ');

    // Build patterns list - start with common patterns, then add extracted keywords
    const patterns: string[] = [];

    // Check if this is a known intent pattern
    if (COMMON_INTENT_PATTERNS[intent.toLowerCase()]) {
      patterns.push(...COMMON_INTENT_PATTERNS[intent.toLowerCase()]);
    } else if (COMMON_INTENT_PATTERNS[normalizedIntent]) {
      patterns.push(...COMMON_INTENT_PATTERNS[normalizedIntent]);
    }

    // Extract keywords from descriptive intent (generalizable)
    const keywords = extractIntentKeywords(intent);
    patterns.push(...keywords);

    // Also add the raw normalized intent as fallback
    if (!patterns.includes(normalizedIntent)) {
      patterns.push(normalizedIntent);
    }

    // Look for elements matching patterns (prefer clickable elements)
    for (const pattern of patterns) {
      for (const element of elements) {
        const text = (element.text || '').toLowerCase();
        const ariaLabel = (element.ariaLabel || '').toLowerCase();
        const name = (element.name || '').toLowerCase();

        if (text.includes(pattern) || ariaLabel.includes(pattern) || name.includes(pattern)) {
          // Prefer clickable buttons/links
          if (element.clickable || element.role === 'button' || element.role === 'link') {
            return element.id;
          }
        }
      }
    }

    // Second pass: less strict matching (any element)
    for (const pattern of patterns) {
      for (const element of elements) {
        const text = (element.text || '').toLowerCase();
        if (text.includes(pattern)) {
          return element.id;
        }
      }
    }

    // Third pass: word-by-word matching for multi-word intents
    // This handles cases where the element text is slightly different
    const intentWords = normalizedIntent.split(/\s+/).filter(w => w.length > 2);
    if (intentWords.length >= 2) {
      for (const element of elements) {
        const text = (element.text || '').toLowerCase();
        const matchCount = intentWords.filter(word => text.includes(word)).length;
        // If more than half the words match, consider it a match
        if (matchCount >= Math.ceil(intentWords.length / 2)) {
          if (element.clickable || element.role === 'button' || element.role === 'link') {
            return element.id;
          }
        }
      }
    }

    return null;
  }

  priorityOrder(): string[] {
    return ['add_to_cart', 'checkout', 'search', 'submit', 'close', 'login'];
  }
}

// ---------------------------------------------------------------------------
// AgentRuntime Interface (minimal for MVP)
// ---------------------------------------------------------------------------

/**
 * Minimal runtime interface for browser control.
 * This will be replaced with the full AgentRuntime integration.
 */
export interface AgentRuntime {
  /** Take a snapshot of the current page */
  snapshot(options?: {
    limit?: number;
    screenshot?: boolean;
    goal?: string;
  }): Promise<Snapshot | null>;

  /** Navigate to a URL */
  goto(url: string): Promise<void>;

  /** Click an element by ID */
  click(elementId: number): Promise<void>;

  /** Type text into an element */
  type(elementId: number, text: string): Promise<void>;

  /** Press a key */
  pressKey(key: string): Promise<void>;

  /** Scroll the page */
  scroll(direction: 'up' | 'down'): Promise<void>;

  /** Get current URL */
  getCurrentUrl(): Promise<string>;

  /** Get viewport height */
  getViewportHeight(): Promise<number>;

  /** Scroll by delta (returns true if scroll was effective) */
  scrollBy(dy: number): Promise<boolean>;
}

// ---------------------------------------------------------------------------
// PlannerExecutorAgent Options
// ---------------------------------------------------------------------------

/**
 * Options for creating a PlannerExecutorAgent.
 */
export interface PlannerExecutorAgentOptions {
  /** LLM for generating plans (recommend 7B+ model) */
  planner: LLMProvider;
  /** LLM for executing steps (3B-7B model) */
  executor: LLMProvider;
  /** Agent configuration (merged with defaults) */
  config?: DeepPartial<PlannerExecutorConfig>;
  /** Pre-action authorization hook */
  preActionAuthorizer?: PreActionAuthorizer;
  /** Custom intent heuristics for element selection */
  intentHeuristics?: IntentHeuristics;
  /** Enable verbose logging */
  verbose?: boolean;
}

// ---------------------------------------------------------------------------
// PlannerExecutorAgent
// ---------------------------------------------------------------------------

/**
 * Two-tier agent architecture with Planner and Executor models.
 *
 * The Planner (typically 7B+ parameters) generates JSON execution plans
 * with predicates. The Executor (3B-7B parameters) executes each step
 * using a snapshot-first approach.
 *
 * @example
 * ```typescript
 * import { PlannerExecutorAgent, OllamaProvider } from '@predicatesystems/runtime';
 *
 * const planner = new OllamaProvider({ model: 'qwen3:8b' });
 * const executor = new OllamaProvider({ model: 'qwen3:4b' });
 *
 * const agent = new PlannerExecutorAgent({
 *   planner,
 *   executor,
 *   config: { stepwise: { maxSteps: 20 }, verbose: true },
 * });
 *
 * const result = await agent.runStepwise(runtime, {
 *   task: 'Search for laptops and add first result to cart',
 *   startUrl: 'https://amazon.com',
 * });
 * ```
 */
export class PlannerExecutorAgent {
  readonly planner: LLMProvider;
  readonly executor: LLMProvider;
  readonly config: PlannerExecutorConfig;

  private preActionAuthorizer?: PreActionAuthorizer;
  private intentHeuristics: IntentHeuristics;
  private tokenCollector = new TokenUsageCollector();

  // Run state
  private runId: string | null = null;
  private actionHistory: ActionRecord[] = [];
  private currentStepIndex = 0;
  private currentStep: { action: string; intent?: string } | null = null;

  constructor(options: PlannerExecutorAgentOptions) {
    this.planner = options.planner;
    this.executor = options.executor;
    this.config = mergeConfig(options.config || {});
    if (options.verbose !== undefined) {
      this.config = { ...this.config, verbose: options.verbose };
    }
    this.preActionAuthorizer = options.preActionAuthorizer;
    this.intentHeuristics = options.intentHeuristics || new SimpleIntentHeuristics();
  }

  // ---------------------------------------------------------------------------
  // Token Stats
  // ---------------------------------------------------------------------------

  /**
   * Get token usage statistics for the agent session.
   */
  getTokenStats(): TokenUsageSummary {
    return this.tokenCollector.summary();
  }

  /**
   * Reset token usage statistics.
   */
  resetTokenStats(): void {
    this.tokenCollector.reset();
  }

  private recordTokenUsage(role: string, resp: LLMResponse): void {
    try {
      this.tokenCollector.record(role, resp);
    } catch {
      // Don't fail on token tracking errors
    }
  }

  // ---------------------------------------------------------------------------
  // Stepwise Run (ReAct-style)
  // ---------------------------------------------------------------------------

  /**
   * Run task using stepwise (ReAct-style) planning.
   *
   * Plans one step at a time based on current page state, adapting to
   * page changes as they happen. More reliable with small models.
   *
   * @param runtime - Browser runtime for page control
   * @param options - Task options
   * @returns Run outcome
   */
  async runStepwise(
    runtime: AgentRuntime,
    options: {
      task: string;
      startUrl?: string;
    }
  ): Promise<RunOutcome> {
    const { task, startUrl } = options;
    const startTime = Date.now();

    // Initialize run state
    this.runId = `run-${Date.now()}-${Math.random().toString(36).slice(2, 8)}`;
    this.actionHistory = [];
    this.currentStepIndex = 0;
    this.tokenCollector.reset();

    const stepOutcomes: StepOutcome[] = [];
    let currentUrl = '';
    let success = false;
    let error: string | undefined;

    try {
      // Navigate to start URL if provided
      if (startUrl) {
        if (this.config.verbose) {
          console.log(`[NAVIGATE] ${startUrl}`);
        }
        await runtime.goto(startUrl);
        currentUrl = startUrl;
      } else {
        currentUrl = await runtime.getCurrentUrl();
      }

      // Stepwise loop
      const maxSteps = this.config.stepwise.maxSteps;

      for (let stepNum = 1; stepNum <= maxSteps; stepNum++) {
        this.currentStepIndex = stepNum;
        const stepStart = Date.now();

        if (this.config.verbose) {
          console.log(`\n${'='.repeat(60)}`);
          console.log(`[STEP ${stepNum}/${maxSteps}]`);
          console.log(`${'='.repeat(60)}`);
        }

        // Take snapshot with escalation
        const ctx = await this.snapshotWithEscalation(runtime, task);
        currentUrl = ctx.snapshot?.url || currentUrl;

        if (this.config.verbose) {
          const elementCount = ctx.snapshot?.elements?.length || 0;
          console.log(`[SNAPSHOT] ${elementCount} elements, limit=${ctx.limitUsed}`);
          // Debug: show searchbox/textbox elements specifically
          const allElements = ctx.snapshot?.elements || [];
          const searchElements = allElements.filter(el =>
            ['textbox', 'searchbox', 'combobox', 'input'].includes((el.role || '').toLowerCase())
          );
          console.log(
            `  [SEARCH ELEMENTS] Found ${searchElements.length} textbox/searchbox/combobox/input elements:`
          );
          for (const el of searchElements.slice(0, 3)) {
            console.log(
              `    [EL ${el.id}] role=${el.role}, text=${(el.text || '').slice(0, 60).replace(/\n/g, ' ')}, clickable=${el.clickable}`
            );
          }
          if (searchElements.length === 0) {
            // Show first 5 elements for debugging
            console.log(`  [FIRST 5 ELEMENTS]:`);
            for (const el of allElements.slice(0, 5)) {
              console.log(
                `    [EL ${el.id}] role=${el.role}, text=${(el.text || '').slice(0, 40).replace(/\n/g, ' ')}`
              );
            }
          }
        }

        // Get planner's next action
        const [systemPrompt, userPrompt] = buildStepwisePlannerPrompt(
          task,
          currentUrl,
          ctx.compactRepresentation,
          this.actionHistory.slice(-this.config.stepwise.actionHistoryLimit)
        );

        if (this.config.verbose) {
          // Show the compact representation being sent to the LLM
          const contextLines = ctx.compactRepresentation.split('\n');
          console.log(`[PLANNER PROMPT] Sending ${contextLines.length} element lines to LLM:`);
          // Show header and first few elements
          for (const line of contextLines.slice(0, 6)) {
            console.log(`  ${line}`);
          }
          if (contextLines.length > 6) {
            console.log(`  ... (${contextLines.length - 6} more elements)`);
          }
        }

        let plannerResp: LLMResponse;
        try {
          plannerResp = await this.planner.generate(systemPrompt, userPrompt, {
            temperature: this.config.plannerTemperature,
            max_tokens: this.config.plannerMaxTokens,
          });
          this.recordTokenUsage('planner', plannerResp);
        } catch (plannerError) {
          // Log planner call failure
          if (this.config.verbose) {
            console.log(`[PLANNER ERROR] LLM call failed: ${plannerError}`);
          }
          stepOutcomes.push({
            stepId: stepNum,
            goal: 'Call planner LLM',
            status: StepStatus.FAILED,
            verificationPassed: false,
            usedVision: false,
            durationMs: Date.now() - stepStart,
            error: `Planner LLM call failed: ${plannerError instanceof Error ? plannerError.message : String(plannerError)}`,
          });
          continue;
        }

        if (this.config.verbose) {
          // Show raw response for debugging (truncated if very long)
          const rawLen = plannerResp.content.length;
          const hasThink = plannerResp.content.includes('<think>');
          const displayContent =
            rawLen > 300
              ? plannerResp.content.slice(0, 300) + `... (${rawLen} chars)`
              : plannerResp.content;
          console.log(`[PLANNER]${hasThink ? ' (has <think>)' : ''} ${displayContent}`);
        }

        // Check for empty response
        if (!plannerResp.content || plannerResp.content.trim().length === 0) {
          if (this.config.verbose) {
            console.log(`[PLANNER ERROR] Empty response from LLM`);
          }
          stepOutcomes.push({
            stepId: stepNum,
            goal: 'Parse planner response',
            status: StepStatus.FAILED,
            verificationPassed: false,
            usedVision: false,
            durationMs: Date.now() - stepStart,
            error: 'Planner returned empty response',
          });
          continue;
        }

        // Parse planner response
        let plannerAction: StepwisePlannerResponse;
        try {
          plannerAction = extractJson(plannerResp.content) as unknown as StepwisePlannerResponse;
        } catch (e) {
          // Try to recover from malformed JSON
          const parsed = parseAction(plannerResp.content);
          if (parsed.action !== 'UNKNOWN') {
            plannerAction = {
              action: parsed.action as StepwisePlannerResponse['action'],
              input: parsed.args[1] as string | undefined,
            };
          } else {
            if (this.config.verbose) {
              console.log(`[PLANNER ERROR] Raw response: ${plannerResp.content.slice(0, 200)}`);
            }
            stepOutcomes.push({
              stepId: stepNum,
              goal: 'Parse planner response',
              status: StepStatus.FAILED,
              verificationPassed: false,
              usedVision: false,
              durationMs: Date.now() - stepStart,
              error: `Failed to parse planner response: ${e}`,
            });
            continue;
          }
        }

        // Handle DONE action
        if (plannerAction.action === 'DONE') {
          if (this.config.verbose) {
            console.log(`[DONE] Task completed`);
          }
          stepOutcomes.push({
            stepId: stepNum,
            goal: 'Task completed',
            status: StepStatus.SUCCESS,
            actionTaken: 'DONE',
            verificationPassed: true,
            usedVision: false,
            durationMs: Date.now() - stepStart,
          });
          success = true;
          break;
        }

        // Execute the action
        const outcome = await this.executeStepwiseAction(
          runtime,
          plannerAction,
          stepNum,
          task,
          ctx,
          stepStart
        );
        stepOutcomes.push(outcome);

        // Record action history
        const urlAfter = await runtime.getCurrentUrl();
        this.actionHistory.push({
          stepNum,
          action: plannerAction.action,
          target: plannerAction.input || plannerAction.intent || null,
          result: outcome.status === StepStatus.SUCCESS ? 'success' : 'failed',
          urlAfter,
        });

        // Update current URL
        currentUrl = urlAfter;

        // Check for repeated failures
        const recentFailures = stepOutcomes.slice(-3).filter(o => o.status === StepStatus.FAILED);
        if (recentFailures.length >= 3) {
          error = 'Too many consecutive failures';
          break;
        }
      }

      // If we ran out of steps without DONE, mark as failed
      if (!success && !error) {
        error = `Exceeded maximum steps (${maxSteps})`;
      }
    } catch (e) {
      error = e instanceof Error ? e.message : String(e);
    }

    return {
      runId: this.runId,
      task,
      success,
      stepsCompleted: stepOutcomes.filter(o => o.status === StepStatus.SUCCESS).length,
      stepsTotal: stepOutcomes.length,
      replansUsed: 0, // Stepwise doesn't use replanning
      stepOutcomes,
      totalDurationMs: Date.now() - startTime,
      error,
      tokenUsage: this.tokenCollector.summary(),
      fallbackUsed: false,
    };
  }

  // ---------------------------------------------------------------------------
  // Execute Stepwise Action
  // ---------------------------------------------------------------------------

  private async executeStepwiseAction(
    runtime: AgentRuntime,
    plannerAction: StepwisePlannerResponse,
    stepNum: number,
    task: string,
    ctx: SnapshotContext,
    stepStart: number
  ): Promise<StepOutcome> {
    const currentUrl = ctx.snapshot?.url || '';

    // Handle SCROLL action
    if (plannerAction.action === 'SCROLL') {
      const direction = plannerAction.direction || 'down';
      try {
        await runtime.scroll(direction);
        return {
          stepId: stepNum,
          goal: `Scroll ${direction}`,
          status: StepStatus.SUCCESS,
          actionTaken: `SCROLL(${direction})`,
          verificationPassed: true,
          usedVision: false,
          durationMs: Date.now() - stepStart,
        };
      } catch (e) {
        return {
          stepId: stepNum,
          goal: `Scroll ${direction}`,
          status: StepStatus.FAILED,
          verificationPassed: false,
          usedVision: false,
          durationMs: Date.now() - stepStart,
          error: e instanceof Error ? e.message : String(e),
        };
      }
    }

    // For CLICK and TYPE_AND_SUBMIT, we need to find the element
    const isTypeAction = plannerAction.action === 'TYPE_AND_SUBMIT';

    // Actions that need to find a target element
    const elementTargetingActions = ['CLICK', 'TYPE_AND_SUBMIT', 'SCROLL_TO'];
    const needsElementLookup = elementTargetingActions.includes(plannerAction.action);

    // Check if target element exists in current snapshot
    // If not, try scroll-after-escalation to find it
    // This is GENERALIZABLE - works for any action that needs to find an element
    let activeCtx = ctx;
    if (needsElementLookup && this.config.snapshot.scrollAfterEscalation && plannerAction.intent) {
      const elements = ctx.snapshot?.elements || [];
      const url = ctx.snapshot?.url || '';
      const foundElement = this.tryIntentHeuristics(plannerAction.intent, elements, url, task);

      if (foundElement === null) {
        // Element not found in current viewport - try scroll-after-escalation
        if (this.config.verbose) {
          console.log(
            `[SCROLL-TO-FIND] Target "${plannerAction.intent}" not in viewport, scrolling to find...`
          );
        }

        const newCtx = await this.snapshotWithEscalation(runtime, task, {
          action: plannerAction.action,
          intent: plannerAction.intent,
        });

        if (newCtx.snapshot) {
          activeCtx = newCtx;
          if (this.config.verbose) {
            console.log(
              `[SCROLL-TO-FIND] Updated context with ${newCtx.snapshot.elements.length} elements`
            );
          }
        }
      }
    }

    const [execSystem, execUser] = buildExecutorPrompt(
      plannerAction.intent || `${plannerAction.action} element`,
      plannerAction.intent,
      activeCtx.compactRepresentation,
      plannerAction.input,
      undefined, // category
      plannerAction.action
    );

    if (this.config.verbose) {
      console.log(`[EXECUTOR PROMPT] system len=${execSystem.length}, user len=${execUser.length}`);
      console.log(`[EXECUTOR USER PROMPT (first 300)]:\n${execUser.slice(0, 300)}...`);
    }

    const executorResp = await this.executor.generate(execSystem, execUser, {
      temperature: this.config.executorTemperature,
      max_tokens: this.config.executorMaxTokens,
    });
    this.recordTokenUsage('executor', executorResp);

    if (this.config.verbose) {
      // Show raw response for debugging (truncated if very long)
      const rawLen = executorResp.content.length;
      const hasThink = executorResp.content.includes('<think>');
      // Show more of the response for debugging
      const displayContent =
        rawLen > 500
          ? executorResp.content.slice(0, 500) + `... (${rawLen} chars total)`
          : executorResp.content;
      console.log(
        `[EXECUTOR RAW]${hasThink ? ' (has <think>)' : ''} len=${rawLen}:\n${displayContent}`
      );
    }

    // Parse executor response
    const parsed = parseAction(executorResp.content);

    // Debug: Show parsed result
    if (this.config.verbose) {
      console.log(`[EXECUTOR PARSED] ${parsed.action}, args: ${JSON.stringify(parsed.args)}`);
    }

    if (parsed.action === 'NONE') {
      return {
        stepId: stepNum,
        goal: plannerAction.intent || plannerAction.action,
        status: StepStatus.FAILED,
        verificationPassed: false,
        usedVision: false,
        durationMs: Date.now() - stepStart,
        error: 'Executor could not find suitable element',
      };
    }

    if (parsed.action === 'UNKNOWN') {
      return {
        stepId: stepNum,
        goal: plannerAction.intent || plannerAction.action,
        status: StepStatus.FAILED,
        verificationPassed: false,
        usedVision: false,
        durationMs: Date.now() - stepStart,
        error: `Failed to parse executor response: ${executorResp.content}`,
      };
    }

    // Pre-action authorization
    if (this.preActionAuthorizer) {
      const actionContext = {
        type: parsed.action,
        elementId: parsed.args[0] as number | undefined,
        value: isTypeAction
          ? ((plannerAction.input || parsed.args[1]) as string | undefined)
          : undefined,
      };
      const authResult = await this.preActionAuthorizer(actionContext, {
        url: currentUrl,
        stepGoal: plannerAction.intent || plannerAction.action,
        taskGoal: task,
      });

      if (!authResult.allowed) {
        return {
          stepId: stepNum,
          goal: plannerAction.intent || plannerAction.action,
          status: StepStatus.FAILED,
          verificationPassed: false,
          usedVision: false,
          durationMs: Date.now() - stepStart,
          error: `Action denied by policy: ${authResult.reason || 'unauthorized'}`,
        };
      }
    }

    // Execute the action
    try {
      const elementId = parsed.args[0] as number;

      if (parsed.action === 'CLICK') {
        await runtime.click(elementId);
        return {
          stepId: stepNum,
          goal: plannerAction.intent || 'Click element',
          status: StepStatus.SUCCESS,
          actionTaken: `CLICK(${elementId})`,
          verificationPassed: true,
          usedVision: false,
          durationMs: Date.now() - stepStart,
          urlBefore: currentUrl,
        };
      } else if (parsed.action === 'TYPE') {
        const text = plannerAction.input || (parsed.args[1] as string) || '';
        await runtime.type(elementId, text);

        // Submit with Enter key for TYPE_AND_SUBMIT
        if (plannerAction.action === 'TYPE_AND_SUBMIT') {
          const preUrl = await runtime.getCurrentUrl();
          let submitMethod: 'enter' | 'click' = 'enter';
          let urlChanged = false;

          // First attempt: Submit with Enter key (more reliable for search)
          await runtime.pressKey('Enter');

          // Wait for URL to change after form submission
          urlChanged = await this.waitForUrlChange(runtime, preUrl, 5000);

          if (this.config.verbose) {
            if (urlChanged) {
              const newUrl = await runtime.getCurrentUrl();
              console.log(`[TYPE_AND_SUBMIT] URL changed after Enter: ${newUrl.slice(0, 60)}...`);
            } else {
              console.log(`[TYPE_AND_SUBMIT] URL unchanged after Enter, attempting retry...`);
            }
          }

          // Retry with button click if Enter didn't work
          if (!urlChanged && this.config.retry.executorRepairAttempts > 0) {
            // Find submit button near the input element
            const submitButtonId = this.findSubmitButton(
              activeCtx.snapshot?.elements || [],
              elementId
            );

            if (submitButtonId !== null) {
              if (this.config.verbose) {
                console.log(
                  `[TYPE_AND_SUBMIT-RETRY] Found submit button ${submitButtonId}, retrying with click`
                );
              }

              try {
                await runtime.click(submitButtonId);
                submitMethod = 'click';
                urlChanged = await this.waitForUrlChange(runtime, preUrl, 5000);

                if (this.config.verbose && urlChanged) {
                  const newUrl = await runtime.getCurrentUrl();
                  console.log(
                    `[TYPE_AND_SUBMIT-RETRY] URL changed after click: ${newUrl.slice(0, 60)}...`
                  );
                }
              } catch (e) {
                if (this.config.verbose) {
                  console.log(`[TYPE_AND_SUBMIT-RETRY] Click failed: ${e}`);
                }
              }
            } else if (this.config.verbose) {
              console.log(`[TYPE_AND_SUBMIT-RETRY] No submit button found for retry`);
            }
          }

          // Wait for page to stabilize
          if (urlChanged) {
            await new Promise(resolve => setTimeout(resolve, 1000));
          }
        }

        return {
          stepId: stepNum,
          goal: plannerAction.intent || 'Type text',
          status: StepStatus.SUCCESS,
          actionTaken: `TYPE(${elementId}, "${text}")`,
          verificationPassed: true,
          usedVision: false,
          durationMs: Date.now() - stepStart,
          urlBefore: currentUrl,
        };
      } else if (parsed.action === 'PRESS') {
        const key = parsed.args[0] as string;
        await runtime.pressKey(key);
        return {
          stepId: stepNum,
          goal: plannerAction.intent || `Press ${key}`,
          status: StepStatus.SUCCESS,
          actionTaken: `PRESS(${key})`,
          verificationPassed: true,
          usedVision: false,
          durationMs: Date.now() - stepStart,
        };
      }

      return {
        stepId: stepNum,
        goal: plannerAction.intent || plannerAction.action,
        status: StepStatus.FAILED,
        verificationPassed: false,
        usedVision: false,
        durationMs: Date.now() - stepStart,
        error: `Unknown action type: ${parsed.action}`,
      };
    } catch (e) {
      return {
        stepId: stepNum,
        goal: plannerAction.intent || plannerAction.action,
        status: StepStatus.FAILED,
        verificationPassed: false,
        usedVision: false,
        durationMs: Date.now() - stepStart,
        error: e instanceof Error ? e.message : String(e),
      };
    }
  }

  // ---------------------------------------------------------------------------
  // Snapshot with Escalation
  // ---------------------------------------------------------------------------

  /**
   * Take snapshot with progressive limit escalation and optional scroll-to-find.
   *
   * Starts with base limit and increases on element-not-found scenarios
   * to capture more of the page. After exhausting limit escalation, if
   * scrollAfterEscalation is enabled, scrolls down/up to find elements
   * that may be outside the current viewport.
   *
   * @param runtime - Browser runtime for snapshots and scrolling
   * @param goal - Goal string for context formatting
   * @param step - Optional step info for intent-based element detection during scroll
   */
  private async snapshotWithEscalation(
    runtime: AgentRuntime,
    goal: string,
    step?: { action: string; intent?: string }
  ): Promise<SnapshotContext> {
    const cfg = this.config.snapshot;
    let currentLimit = cfg.limitBase;
    const maxLimit = cfg.enabled ? cfg.limitMax : cfg.limitBase;
    let lastSnapshot: Awaited<ReturnType<AgentRuntime['snapshot']>> = null;
    let lastCompact = '';
    let requiresVision = false;
    let visionReason: string | null = null;

    // Phase 1: Limit escalation loop
    while (currentLimit <= maxLimit) {
      try {
        const snap = await runtime.snapshot({
          limit: currentLimit,
          screenshot: false,
          goal,
        });

        if (snap === null) {
          if (!cfg.enabled) break;
          currentLimit = Math.min(currentLimit + cfg.limitStep, maxLimit + 1);
          continue;
        }

        lastSnapshot = snap;
        lastCompact = formatContext(snap.elements || [], currentLimit);

        // If escalation disabled, we're done after first successful snapshot
        if (!cfg.enabled) break;

        // Check element count - if sufficient, no need to escalate
        const elementCount = snap.elements?.length || 0;
        if (elementCount >= 10) break;

        // Escalate limit
        if (currentLimit < maxLimit) {
          currentLimit = Math.min(currentLimit + cfg.limitStep, maxLimit);
          if (this.config.verbose) {
            console.log(
              `[ESCALATION] Low element count (${elementCount}), increasing limit to ${currentLimit}`
            );
          }
        } else {
          break;
        }
      } catch (e) {
        if (this.config.verbose) {
          console.error(`[SNAPSHOT] Error: ${e}`);
        }
        if (!cfg.enabled) break;
        currentLimit = Math.min(currentLimit + cfg.limitStep, maxLimit + 1);
      }
    }

    // Phase 2: Scroll-after-escalation
    // Only trigger for CLICK actions with specific intents
    const shouldTryScroll =
      cfg.scrollAfterEscalation &&
      step !== undefined &&
      lastSnapshot !== null &&
      !requiresVision &&
      step.action === 'CLICK' &&
      step.intent;

    if (shouldTryScroll && lastSnapshot) {
      // Check if we can find the target element using intent heuristics
      const elements = lastSnapshot.elements || [];
      const url = lastSnapshot.url || '';
      let foundElement = this.tryIntentHeuristics(step.intent!, elements, url, goal);

      if (foundElement === null) {
        // Element not found in current viewport - try scrolling
        if (this.config.verbose) {
          console.log(
            `[SNAPSHOT-ESCALATION] Target element not found for intent "${step.intent}", trying scroll-after-escalation...`
          );
        }

        // Get viewport height and calculate scroll delta
        const viewportHeight = await runtime.getViewportHeight();
        const scrollDelta = viewportHeight * cfg.scrollViewportFraction;

        for (const direction of cfg.scrollDirections) {
          // Map direction to dy (pixels): down=positive, up=negative
          const scrollDy = direction === 'down' ? scrollDelta : -scrollDelta;

          for (let scrollNum = 0; scrollNum < cfg.scrollMaxAttempts; scrollNum++) {
            if (this.config.verbose) {
              console.log(
                `[SNAPSHOT-ESCALATION] Scrolling ${direction} (${scrollNum + 1}/${cfg.scrollMaxAttempts})...`
              );
            }

            // Scroll with verification
            const scrollEffective = await runtime.scrollBy(scrollDy);

            if (!scrollEffective) {
              if (this.config.verbose) {
                console.log(
                  `[SNAPSHOT-ESCALATION] Scroll ${direction} had no effect (reached boundary), skipping remaining attempts`
                );
              }
              break; // No point trying more scrolls in this direction
            }

            // Wait for stabilization after successful scroll
            if (cfg.scrollStabilizeMs > 0) {
              await new Promise(resolve => setTimeout(resolve, cfg.scrollStabilizeMs));
            }

            // Take new snapshot at max limit (we already escalated)
            try {
              const snap = await runtime.snapshot({
                limit: cfg.limitMax,
                screenshot: false,
                goal,
              });

              if (snap === null) continue;

              lastSnapshot = snap;
              lastCompact = formatContext(snap.elements || [], cfg.limitMax);

              // Check if target element is now visible
              const newElements = snap.elements || [];
              const newUrl = snap.url || '';
              foundElement = this.tryIntentHeuristics(step.intent!, newElements, newUrl, goal);

              if (foundElement !== null) {
                if (this.config.verbose) {
                  console.log(
                    `[SNAPSHOT-ESCALATION] Found target element ${foundElement} after scrolling ${direction}`
                  );
                }
                break; // Break out of scroll attempts loop
              }
            } catch {
              continue;
            }
          }

          // If found, break out of direction loop
          if (foundElement !== null) break;
        }

        if (foundElement === null && this.config.verbose) {
          console.log(`[SNAPSHOT-ESCALATION] Target element not found after scrolling`);
        }
      }
    }

    // Fallback for failed capture
    if (lastSnapshot === null) {
      lastSnapshot = { url: '', title: '', elements: [] };
      requiresVision = true;
      visionReason = 'snapshot_capture_failed';
    }

    return {
      snapshot: lastSnapshot,
      compactRepresentation: lastCompact,
      screenshotBase64: null,
      capturedAt: new Date(),
      limitUsed: currentLimit,
      snapshotSuccess: !requiresVision,
      requiresVision,
      visionReason,
      pruningCategory: null,
      prunedNodeCount: 0,
    };
  }

  // ---------------------------------------------------------------------------
  // Intent Heuristics Helper
  // ---------------------------------------------------------------------------

  /**
   * Try to find an element using intent heuristics.
   *
   * @param intent - Intent string (e.g., "add_to_cart")
   * @param elements - Available elements from snapshot
   * @param url - Current page URL
   * @param goal - Step goal description
   * @returns Element ID if found, null otherwise
   */
  private tryIntentHeuristics(
    intent: string,
    elements: SnapshotElement[],
    url: string,
    goal: string
  ): number | null {
    try {
      return this.intentHeuristics.findElementForIntent(intent, elements, url, goal);
    } catch {
      return null;
    }
  }

  // ---------------------------------------------------------------------------
  // Retry/Repair Helpers
  // ---------------------------------------------------------------------------

  /**
   * Wait for URL to change from the given URL.
   *
   * @param runtime - Browser runtime
   * @param originalUrl - URL to compare against
   * @param timeoutMs - Maximum wait time in milliseconds
   * @returns true if URL changed, false if timeout
   */
  private async waitForUrlChange(
    runtime: AgentRuntime,
    originalUrl: string,
    timeoutMs: number
  ): Promise<boolean> {
    const startTime = Date.now();
    const pollInterval = 500;

    while (Date.now() - startTime < timeoutMs) {
      await new Promise(resolve => setTimeout(resolve, pollInterval));
      try {
        const currentUrl = await runtime.getCurrentUrl();
        if (currentUrl !== originalUrl) {
          return true;
        }
      } catch {
        // Ignore errors during URL check
      }
    }
    return false;
  }

  /**
   * Find a submit button near the input element.
   *
   * Looks for buttons/links with submit-related text that appear after
   * the input element in the DOM (higher element ID typically means
   * later in DOM order).
   *
   * @param elements - Snapshot elements
   * @param inputElementId - ID of the input element
   * @returns Submit button element ID if found, null otherwise
   */
  private findSubmitButton(elements: SnapshotElement[], inputElementId: number): number | null {
    // Submit-related patterns
    const submitPatterns = [
      'search',
      'go',
      'find',
      'submit',
      'send',
      'enter',
      'apply',
      'ok',
      'done',
    ];

    // Icon patterns (exact match)
    const iconPatterns = ['>', '→', '🔍', '⌕'];

    // Look for submit buttons
    const candidates: Array<{ id: number; score: number }> = [];

    for (const element of elements) {
      // Only consider buttons and links
      const role = (element.role || '').toLowerCase();
      if (!['button', 'link', 'searchbox'].includes(role)) continue;

      // Skip if not clickable
      if (element.clickable === false) continue;

      // Skip the input element itself
      if (element.id === inputElementId) continue;

      const text = (element.text || '').toLowerCase().trim();
      const ariaLabel = (element.ariaLabel || '').toLowerCase();

      // Check for icon patterns (exact match, high priority)
      for (const icon of iconPatterns) {
        if (text === icon || ariaLabel === icon) {
          candidates.push({ id: element.id, score: 200 + Math.abs(element.id - inputElementId) });
          break;
        }
      }

      // Check for submit patterns
      for (let i = 0; i < submitPatterns.length; i++) {
        const pattern = submitPatterns[i];
        if (text.includes(pattern) || ariaLabel.includes(pattern)) {
          // Score: pattern priority + proximity to input element
          // Lower distance from input = higher score
          const proximityBonus = 100 - Math.min(Math.abs(element.id - inputElementId), 100);
          candidates.push({ id: element.id, score: 100 - i + proximityBonus });
          break;
        }
      }
    }

    // Return best candidate (highest score)
    if (candidates.length === 0) return null;
    candidates.sort((a, b) => b.score - a.score);
    return candidates[0].id;
  }
}
