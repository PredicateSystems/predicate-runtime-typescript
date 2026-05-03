import type { RecoveryNavigationConfig } from './recovery';
import { DEFAULT_RECOVERY_CONFIG } from './recovery';
import type { ModalDismissalConfig } from './modal-dismissal';
import { DEFAULT_MODAL_CONFIG } from './modal-dismissal';
import type { AuthBoundaryConfig } from './boundary-detection';
import { DEFAULT_AUTH_BOUNDARY_CONFIG } from './boundary-detection';

/**
 * PlannerExecutorAgent Configuration
 *
 * Configuration interfaces and presets for the planner-executor agent architecture.
 */

/**
 * Snapshot escalation configuration for reliable element capture.
 *
 * When element selection fails, the agent can retry with increasing element limits.
 * After exhausting limit escalation, scroll-after-escalation can be used to find
 * elements that may be outside the current viewport.
 *
 * @example
 * ```typescript
 * // Default: escalation enabled with step=30
 * const config: SnapshotEscalationConfig = { enabled: true, limitBase: 60, limitStep: 30 };
 *
 * // Enable scroll-after-escalation to find elements below/above viewport
 * const config: SnapshotEscalationConfig = {
 *   ...DEFAULT_CONFIG.snapshot,
 *   scrollAfterEscalation: true,
 *   scrollDirections: ['down', 'up'],
 * };
 * ```
 */
export interface SnapshotEscalationConfig {
  /** Whether escalation is enabled (default: true) */
  enabled: boolean;
  /** Starting element limit (default: 60) */
  limitBase: number;
  /** Increase per escalation step (default: 30) */
  limitStep: number;
  /** Maximum element limit (default: 200) */
  limitMax: number;
  /** Whether to scroll after limit escalation is exhausted (default: true) */
  scrollAfterEscalation: boolean;
  /** Maximum scroll attempts per direction (default: 3) */
  scrollMaxAttempts: number;
  /** Directions to try scrolling (default: ['down', 'up']) */
  scrollDirections: Array<'up' | 'down'>;
  /** Scroll amount as fraction of viewport height (default: 0.4 = 40%) */
  scrollViewportFraction: number;
  /** Stabilization delay after scroll in ms (default: 300) */
  scrollStabilizeMs: number;
  /** Whether category-specific pruning is enabled for compact context (default: true) */
  pruningEnabled: boolean;
  /** Minimum retained elements before pruning relaxes automatically (default: 3) */
  pruningMinElements: number;
  /** Maximum pruning relaxation level before falling back to generic context (default: 2) */
  pruningMaxRelaxation: number;
}

/**
 * Retry and verification configuration.
 */
export interface RetryConfig {
  /** Verification timeout in milliseconds (default: 10000) */
  verifyTimeoutMs: number;
  /** Verification poll interval in milliseconds (default: 500) */
  verifyPollMs: number;
  /** Maximum verification attempts (default: 4) */
  verifyMaxAttempts: number;
  /** Executor repair attempts on action failure (default: 2) */
  executorRepairAttempts: number;
  /** Maximum replan attempts (default: 2) */
  maxReplans: number;
  /** Base delay in ms for LLM 429 rate-limit retries (default: 5000) */
  llmRetryBackoffMs: number;
  /** Maximum retries for LLM 429 errors before giving up (default: 3) */
  llmMaxRetries: number;
}

/**
 * Stepwise (ReAct-style) planning configuration.
 */
export interface StepwisePlanningConfig {
  /** Maximum steps per run (default: 20) */
  maxSteps: number;
  /** Number of recent actions to include in context (default: 5) */
  actionHistoryLimit: number;
  /** Whether to include page title/URL in context (default: true) */
  includePageContext: boolean;
}

/**
 * Full configuration for PlannerExecutorAgent.
 */
export interface PlannerExecutorConfig {
  /** Snapshot escalation settings */
  snapshot: SnapshotEscalationConfig;

  /** Retry and verification settings */
  retry: RetryConfig;

  /** Stepwise planning settings */
  stepwise: StepwisePlanningConfig;

  /** Maximum tokens for planner LLM (default: 2048) */
  plannerMaxTokens: number;

  /** Temperature for planner LLM (default: 0.0) */
  plannerTemperature: number;

  /** Maximum tokens for executor LLM (default: 96) */
  executorMaxTokens: number;

  /** Temperature for executor LLM (default: 0.0) */
  executorTemperature: number;

  /** Whether to check predicates before step execution (default: true) */
  preStepVerification: boolean;

  /** Recovery navigation settings */
  recovery: RecoveryNavigationConfig;

  /** Automatic modal/drawer dismissal settings */
  modal: ModalDismissalConfig;

  /** Authentication boundary detection settings */
  authBoundary: AuthBoundaryConfig;

  /** Whether to enable verbose logging (default: false) */
  verbose: boolean;
}

/**
 * Default configuration values.
 */
export const DEFAULT_CONFIG: PlannerExecutorConfig = {
  snapshot: {
    enabled: true,
    // Same defaults as Python SDK - formatContext uses multi-strategy selection
    // to ensure product links are captured even with lower snapshot limits
    limitBase: 60, // Initial snapshot limit (Python SDK default)
    limitStep: 30, // Escalation step (Python SDK default)
    limitMax: 200, // Maximum limit (Python SDK default)
    scrollAfterEscalation: true,
    scrollMaxAttempts: 3,
    scrollDirections: ['down', 'up'],
    scrollViewportFraction: 0.4,
    scrollStabilizeMs: 300,
    pruningEnabled: true,
    pruningMinElements: 3,
    pruningMaxRelaxation: 2,
  },
  retry: {
    verifyTimeoutMs: 10000,
    verifyPollMs: 500,
    verifyMaxAttempts: 4,
    executorRepairAttempts: 2,
    maxReplans: 2,
    llmRetryBackoffMs: 5000,
    llmMaxRetries: 3,
  },
  stepwise: {
    maxSteps: 20,
    actionHistoryLimit: 5,
    includePageContext: true,
  },
  plannerMaxTokens: 2048,
  plannerTemperature: 0.0,
  executorMaxTokens: 96,
  executorTemperature: 0.0,
  preStepVerification: true,
  recovery: { ...DEFAULT_RECOVERY_CONFIG },
  modal: { ...DEFAULT_MODAL_CONFIG },
  authBoundary: { ...DEFAULT_AUTH_BOUNDARY_CONFIG },
  verbose: false,
};

/**
 * Pre-configured settings for common use cases.
 */
export enum ConfigPreset {
  /** Default balanced configuration */
  DEFAULT = 'default',
  /** Optimized for 4B-8B local models (Ollama) */
  LOCAL_SMALL_MODEL = 'local_small',
  /** Optimized for high-capability cloud models (GPT-4, Claude) */
  CLOUD_HIGH_QUALITY = 'cloud_high',
  /** Minimal retries for rapid development */
  FAST_ITERATION = 'fast',
  /** Conservative settings for production reliability */
  PRODUCTION = 'production',
}

/**
 * Get a pre-configured PlannerExecutorConfig for common use cases.
 *
 * @param preset - Preset name or ConfigPreset enum value
 * @returns PlannerExecutorConfig with preset values
 *
 * @example
 * ```typescript
 * import { getConfigPreset, ConfigPreset } from '@predicatesystems/runtime';
 *
 * const config = getConfigPreset(ConfigPreset.LOCAL_SMALL_MODEL);
 * ```
 */
export function getConfigPreset(preset: ConfigPreset | string): PlannerExecutorConfig {
  // Normalize to string for comparison (enum values are strings)
  const presetKey: string = typeof preset === 'string' ? preset : (preset as string);

  switch (presetKey) {
    case ConfigPreset.LOCAL_SMALL_MODEL as string:
    case 'local_small':
      // Optimized for local 4B-8B models (Ollama)
      // - Higher token limits for models like Qwen3 that include reasoning in output
      // - More lenient timeouts for slower local inference
      // - Higher element limits to capture product links on e-commerce pages
      // - Verbose mode helpful for debugging local model behavior
      return {
        ...DEFAULT_CONFIG,
        snapshot: {
          ...DEFAULT_CONFIG.snapshot,
          // Higher limits needed for e-commerce - many elements filtered to interactive roles
          limitBase: 200, // Capture more elements (was 60)
          limitStep: 50, // Larger escalation steps (was 30)
          limitMax: 400, // Higher max for complex pages (was 200)
        },
        retry: {
          ...DEFAULT_CONFIG.retry,
          verifyTimeoutMs: 15000,
          verifyMaxAttempts: 6,
          executorRepairAttempts: 3,
          maxReplans: 2,
        },
        // No token limit for planner - let Qwen3 thinking models complete reasoning
        // Small local models need room to think through the task step by step
        plannerMaxTokens: 8192,
        // Higher token limit to accommodate Qwen3/DeepSeek models that output reasoning
        // before the actual action. Qwen3 models can use 4000+ chars of reasoning before
        // outputting the actual action. Need enough headroom for the model to complete.
        executorMaxTokens: 4096,
        verbose: true,
      };

    case ConfigPreset.CLOUD_HIGH_QUALITY as string:
    case 'cloud_high':
      // Optimized for high-capability cloud models (GPT-4, Claude)
      // - Higher token limits for more detailed plans
      // - Faster timeouts (cloud inference is quick)
      // - Verbose off for cleaner output
      return {
        ...DEFAULT_CONFIG,
        retry: {
          ...DEFAULT_CONFIG.retry,
          verifyTimeoutMs: 10000,
          verifyMaxAttempts: 4,
          executorRepairAttempts: 2,
          maxReplans: 2,
        },
        plannerMaxTokens: 2048,
        executorMaxTokens: 128,
        verbose: false,
      };

    case ConfigPreset.FAST_ITERATION as string:
    case 'fast':
      // For rapid development and testing
      // - Minimal retries to fail fast
      // - Verbose for debugging
      return {
        ...DEFAULT_CONFIG,
        retry: {
          ...DEFAULT_CONFIG.retry,
          verifyTimeoutMs: 5000,
          verifyMaxAttempts: 2,
          executorRepairAttempts: 1,
          maxReplans: 1,
        },
        plannerMaxTokens: 1024,
        executorMaxTokens: 64,
        verbose: true,
      };

    case ConfigPreset.PRODUCTION as string:
    case 'production':
      // Conservative settings for production reliability
      // - More retries for robustness
      // - Longer timeouts for edge cases
      // - No verbose output
      return {
        ...DEFAULT_CONFIG,
        retry: {
          ...DEFAULT_CONFIG.retry,
          verifyTimeoutMs: 20000,
          verifyMaxAttempts: 8,
          executorRepairAttempts: 3,
          maxReplans: 3,
        },
        plannerMaxTokens: 2048,
        executorMaxTokens: 128,
        verbose: false,
      };

    case ConfigPreset.DEFAULT as string:
    case 'default':
    default:
      return { ...DEFAULT_CONFIG };
  }
}

/**
 * Deep partial type for nested configuration.
 */
export type DeepPartial<T> = {
  [P in keyof T]?: T[P] extends object ? DeepPartial<T[P]> : T[P];
};

/**
 * Merge partial config with defaults.
 *
 * @param partial - Partial configuration to merge
 * @returns Complete PlannerExecutorConfig
 */
export function mergeConfig(partial: DeepPartial<PlannerExecutorConfig>): PlannerExecutorConfig {
  const snapshot: SnapshotEscalationConfig = {
    ...DEFAULT_CONFIG.snapshot,
    ...(partial.snapshot ?? {}),
    // Ensure scrollDirections has correct type
    scrollDirections: (partial.snapshot?.scrollDirections ??
      DEFAULT_CONFIG.snapshot.scrollDirections) as Array<'up' | 'down'>,
  };
  const modal: ModalDismissalConfig = {
    ...DEFAULT_CONFIG.modal,
    ...(partial.modal ?? {}),
    roleFilter: (partial.modal?.roleFilter ?? DEFAULT_CONFIG.modal.roleFilter) as string[],
    dismissPatterns: (partial.modal?.dismissPatterns ??
      DEFAULT_CONFIG.modal.dismissPatterns) as string[],
    iconPatterns: (partial.modal?.iconPatterns ?? DEFAULT_CONFIG.modal.iconPatterns) as string[],
    checkoutPatterns: (partial.modal?.checkoutPatterns ??
      DEFAULT_CONFIG.modal.checkoutPatterns) as string[],
  };
  const authBoundary: AuthBoundaryConfig = {
    ...DEFAULT_CONFIG.authBoundary,
    ...(partial.authBoundary ?? {}),
    urlPatterns: (partial.authBoundary?.urlPatterns ??
      DEFAULT_CONFIG.authBoundary.urlPatterns) as string[],
  };

  return {
    ...DEFAULT_CONFIG,
    ...partial,
    snapshot,
    retry: { ...DEFAULT_CONFIG.retry, ...(partial.retry ?? {}) },
    stepwise: { ...DEFAULT_CONFIG.stepwise, ...(partial.stepwise ?? {}) },
    recovery: { ...DEFAULT_CONFIG.recovery, ...(partial.recovery ?? {}) },
    modal,
    authBoundary,
  };
}
