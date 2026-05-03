import type { Snapshot, StepHookContext } from '../types';
import type { PermissionPolicy } from '../browser';
import type { AgentRuntime } from '../agent-runtime';
import { LLMProvider } from '../llm-provider';
import { RuntimeAgent } from '../runtime-agent';
import type { RuntimeStep } from '../runtime-agent';
import type { CaptchaOptions } from '../captcha/types';
import type { CaptchaHandler } from '../captcha/types';

export interface PermissionRecoveryConfig {
  enabled?: boolean;
  maxRestarts?: number;
  autoGrant?: string[];
  geolocation?: Record<string, any> | null;
  origin?: string | null;
}

export interface VisionFallbackConfig {
  enabled?: boolean;
  maxVisionCalls?: number;
  triggerRequiresVision?: boolean;
  triggerRepeatedNoop?: boolean;
  triggerCanvasOrLowActionables?: boolean;
}

export interface CaptchaConfig {
  policy?: 'abort' | 'callback';
  // Interface-only: SDK does not ship captcha solvers. Users provide a handler/callback.
  handler?: CaptchaHandler | null;
  timeoutMs?: number | null;
  pollMs?: number | null;
  minConfidence?: number;
}

export interface PredicateBrowserAgentConfig {
  // Permissions
  permissionStartup?: PermissionPolicy | null;
  permissionRecovery?: PermissionRecoveryConfig | null;

  // Vision fallback
  vision?: VisionFallbackConfig;

  // CAPTCHA handling
  captcha?: CaptchaConfig;

  // Prompt / token controls
  historyLastN?: number; // 0 disables LLM-facing step history

  // Opt-in: track token usage from LLM provider responses (best-effort).
  tokenUsageEnabled?: boolean;

  // Compact prompt customization
  // builder(taskGoal, stepGoal, domContext, snapshot, historySummary) -> {systemPrompt, userPrompt}
  compactPromptBuilder?: (
    taskGoal: string,
    stepGoal: string,
    domContext: string,
    snap: Snapshot,
    historySummary: string
  ) => { systemPrompt: string; userPrompt: string };

  compactPromptPostprocessor?: (domContext: string) => string;
}

function historySummary(items: string[]): string {
  if (!items.length) return '';
  return items.map(s => `- ${s}`).join('\n');
}

function applyCaptchaConfigToRuntime(runtime: AgentRuntime, cfg: CaptchaConfig | undefined): void {
  if (!cfg) return;

  const policy = (cfg.policy ?? 'abort').toLowerCase() as 'abort' | 'callback';
  if (policy === 'abort') {
    runtime.setCaptchaOptions({
      policy: 'abort',
      minConfidence: cfg.minConfidence ?? 0.7,
    } satisfies CaptchaOptions);
    return;
  }

  const pollMs = cfg.pollMs ?? 1_000;
  const timeoutMs = cfg.timeoutMs ?? 120_000;
  const minConfidence = cfg.minConfidence ?? 0.7;

  const handler = cfg.handler ?? null;
  if (!handler) {
    throw new Error(
      'captcha.handler is required when captcha.policy="callback". ' +
        'Provide a handler callback (e.g. human handoff or your external system).'
    );
  }

  runtime.setCaptchaOptions({
    policy: 'callback',
    handler,
    timeoutMs,
    pollMs,
    minConfidence,
  } satisfies CaptchaOptions);
}

type TokenUsageTotals = {
  calls: number;
  promptTokens: number;
  completionTokens: number;
  totalTokens: number;
};

class TokenUsageCollector {
  private byRole: Record<string, TokenUsageTotals> = {};
  private byModel: Record<string, TokenUsageTotals> = {};

  record(role: string, resp: any): void {
    const pt = typeof resp?.promptTokens === 'number' ? resp.promptTokens : 0;
    const ct = typeof resp?.completionTokens === 'number' ? resp.completionTokens : 0;
    const tt = typeof resp?.totalTokens === 'number' ? resp.totalTokens : pt + ct;
    const model = String(resp?.modelName ?? 'unknown') || 'unknown';

    const bump = (dst: Record<string, TokenUsageTotals>, key: string) => {
      const cur =
        dst[key] ??
        ({ calls: 0, promptTokens: 0, completionTokens: 0, totalTokens: 0 } as TokenUsageTotals);
      cur.calls += 1;
      cur.promptTokens += Math.max(0, pt);
      cur.completionTokens += Math.max(0, ct);
      cur.totalTokens += Math.max(0, tt);
      dst[key] = cur;
    };

    bump(this.byRole, role);
    bump(this.byModel, model);
  }

  reset(): void {
    this.byRole = {};
    this.byModel = {};
  }

  summary(): {
    total: TokenUsageTotals;
    byRole: Record<string, TokenUsageTotals>;
    byModel: Record<string, TokenUsageTotals>;
  } {
    const sum = (src: Record<string, TokenUsageTotals>): TokenUsageTotals => {
      return Object.values(src).reduce(
        (acc, v) => ({
          calls: acc.calls + v.calls,
          promptTokens: acc.promptTokens + v.promptTokens,
          completionTokens: acc.completionTokens + v.completionTokens,
          totalTokens: acc.totalTokens + v.totalTokens,
        }),
        { calls: 0, promptTokens: 0, completionTokens: 0, totalTokens: 0 }
      );
    };
    return { total: sum(this.byRole), byRole: this.byRole, byModel: this.byModel };
  }
}

class TokenAccountingProvider extends LLMProvider {
  constructor(
    private inner: LLMProvider,
    private collector: TokenUsageCollector,
    private role: string
  ) {
    super();
  }
  get modelName(): string {
    return this.inner.modelName;
  }
  supportsJsonMode(): boolean {
    return this.inner.supportsJsonMode();
  }
  supportsVision(): boolean {
    return this.inner.supportsVision?.() ?? false;
  }
  async generate(
    systemPrompt: string,
    userPrompt: string,
    options: Record<string, any> = {}
  ): Promise<any> {
    const resp = await this.inner.generate(systemPrompt, userPrompt, options);
    try {
      this.collector.record(this.role, resp);
    } catch {
      // best-effort
    }
    return resp;
  }
  async generateWithImage(
    systemPrompt: string,
    userPrompt: string,
    imageBase64: string,
    options: Record<string, any> = {}
  ): Promise<any> {
    const fn = (this.inner as any).generateWithImage;
    if (typeof fn !== 'function') {
      throw new Error('Inner provider does not implement generateWithImage');
    }
    const resp = await fn.call(this.inner, systemPrompt, userPrompt, imageBase64, options);
    try {
      this.collector.record(this.role, resp);
    } catch {
      // best-effort
    }
    return resp;
  }
}

// Re-export planner-executor profile/learning types for extension consumption
export type {
  DataDrivenPruningPolicy,
  BrowserAgentProfile,
  ResolvedAgentProfile,
  LearnedTargetFingerprint,
  DomainProfile,
} from './planner-executor/profile-types';
export { EMPTY_RESOLVED_PROFILE } from './planner-executor/profile-types';
export {
  DataDrivenPruningPolicySchema,
  BrowserAgentProfileSchema,
  BrowserAgentProfileArraySchema,
  LearnedTargetFingerprintSchema,
  DomainProfileSchema,
} from './planner-executor/profile-schema';
export { ProfileRegistry } from './planner-executor/profile-registry';
export { pruneWithPolicy } from './planner-executor/data-driven-pruner';
export {
  computeTaskHash,
  extractDomain,
  createFingerprint,
  mergeFingerprint,
  recordFingerprintFailure,
} from './planner-executor/fingerprint-normalizer';
export type { LearningStore } from './planner-executor/learning-store';
export { InMemoryLearningStore } from './planner-executor/learning-store';
export {
  extractFingerprintFromOutcome,
  applyFingerprintFailure,
  applyFingerprintSuccess,
  isFingerprintStale,
  isFingerprintExpired,
  fingerprintToHint,
  computeTaskHash as computeAsyncTaskHash,
  isSensitiveUrl as isLearningSensitiveUrl,
} from './planner-executor/learning-extractor';
export type {
  LearningExtractionOptions,
  LearningExtractionResult,
} from './planner-executor/learning-extractor';

export type StepOutcome = { stepGoal: string; ok: boolean };

export class PredicateBrowserAgent {
  readonly runtime: AgentRuntime;
  readonly executor: LLMProvider;
  readonly visionExecutor?: LLMProvider;
  readonly visionVerifier?: LLMProvider;
  readonly config: PredicateBrowserAgentConfig;

  private history: string[] = [];
  private visionCallsUsed = 0;
  private runner: RuntimeAgent;
  private tokenUsage: TokenUsageCollector | null = null;

  constructor(opts: {
    runtime: AgentRuntime;
    executor: LLMProvider;
    visionExecutor?: LLMProvider;
    visionVerifier?: LLMProvider;
    config?: PredicateBrowserAgentConfig;
  }) {
    const tokenUsageEnabled = Boolean(opts.config?.tokenUsageEnabled);
    const collector = tokenUsageEnabled ? new TokenUsageCollector() : null;

    this.runtime = opts.runtime;
    this.tokenUsage = collector;
    this.executor = collector
      ? new TokenAccountingProvider(opts.executor, collector, 'executor')
      : opts.executor;
    this.visionExecutor =
      collector && opts.visionExecutor
        ? new TokenAccountingProvider(opts.visionExecutor, collector, 'vision_executor')
        : opts.visionExecutor;
    this.visionVerifier =
      collector && opts.visionVerifier
        ? new TokenAccountingProvider(opts.visionVerifier, collector, 'vision_verifier')
        : opts.visionVerifier;
    this.config = {
      permissionStartup: null,
      permissionRecovery: null,
      vision: { enabled: false, maxVisionCalls: 0 },
      captcha: { policy: 'abort', handler: null },
      historyLastN: 0,
      ...(opts.config ?? {}),
    };

    applyCaptchaConfigToRuntime(this.runtime, this.config.captcha);

    this.runner = new RuntimeAgent({
      runtime: this.runtime,
      executor: this.executor,
      visionExecutor: this.visionExecutor,
      visionVerifier: this.visionVerifier,
      structuredPromptBuilder: this.config.compactPromptBuilder,
      domContextPostprocessor: this.config.compactPromptPostprocessor,
      historySummaryProvider: () => {
        const n = Math.max(0, this.config.historyLastN ?? 0);
        if (n <= 0) return '';
        const slice = this.history.slice(Math.max(0, this.history.length - n));
        return historySummary(slice);
      },
    } as any);
  }

  getTokenUsage(): any {
    if (!this.tokenUsage) {
      return { enabled: false, reason: 'tokenUsageEnabled is false' };
    }
    return { enabled: true, ...this.tokenUsage.summary() };
  }

  resetTokenUsage(): void {
    this.tokenUsage?.reset();
  }

  private recordHistory(stepGoal: string, ok: boolean) {
    const n = Math.max(0, this.config.historyLastN ?? 0);
    if (n <= 0) return;
    this.history.push(`${stepGoal} -> ${ok ? 'ok' : 'fail'}`);
    if (this.history.length > n) {
      this.history = this.history.slice(this.history.length - n);
    }
  }

  async step(opts: {
    taskGoal: string;
    step: RuntimeStep;
    onStepStart?: (ctx: StepHookContext) => void | Promise<void>;
    onStepEnd?: (ctx: StepHookContext) => void | Promise<void>;
  }): Promise<StepOutcome> {
    let step = opts.step;

    const maxVisionCalls = Math.max(0, this.config.vision?.maxVisionCalls ?? 0);
    if (
      this.config.vision?.enabled &&
      maxVisionCalls > 0 &&
      this.visionCallsUsed >= maxVisionCalls
    ) {
      step = { ...step, visionExecutorEnabled: false, maxVisionExecutorAttempts: 0 };
    }

    const ok = await this.runner.runStep({
      taskGoal: opts.taskGoal,
      step,
      onStepStart: opts.onStepStart,
      onStepEnd: opts.onStepEnd,
    });

    this.recordHistory(step.goal, ok);
    return { stepGoal: step.goal, ok };
  }

  async run(opts: {
    taskGoal: string;
    steps: RuntimeStep[];
    onStepStart?: (ctx: StepHookContext) => void | Promise<void>;
    onStepEnd?: (ctx: StepHookContext) => void | Promise<void>;
    stopOnFailure?: boolean;
  }): Promise<boolean> {
    const stopOnFailure = opts.stopOnFailure ?? true;
    for (const step of opts.steps) {
      const out = await this.step({
        taskGoal: opts.taskGoal,
        step,
        onStepStart: opts.onStepStart,
        onStepEnd: opts.onStepEnd,
      });
      if (stopOnFailure && !out.ok) return false;
    }
    return true;
  }
}
