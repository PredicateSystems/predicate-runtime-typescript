/**
 * PlannerExecutorAgent Configuration
 *
 * Configuration interfaces and presets for the planner-executor agent architecture.
 */

/**
 * Snapshot escalation configuration for reliable element capture.
 *
 * When element selection fails, the agent can retry with increasing element limits.
 */
export interface SnapshotEscalationConfig {
  /** Whether escalation is enabled (default: true) */
  enabled: boolean;
  /** Starting element limit (default: 50) */
  limitBase: number;
  /** Increase per escalation step (default: 25) */
  limitStep: number;
  /** Maximum element limit (default: 200) */
  limitMax: number;
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

  /** Whether to enable verbose logging (default: false) */
  verbose: boolean;
}

/**
 * Default configuration values.
 */
export const DEFAULT_CONFIG: PlannerExecutorConfig = {
  snapshot: {
    enabled: true,
    limitBase: 50,
    limitStep: 25,
    limitMax: 200,
  },
  retry: {
    verifyTimeoutMs: 10000,
    verifyPollMs: 500,
    verifyMaxAttempts: 4,
    executorRepairAttempts: 2,
    maxReplans: 2,
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
      // - Tighter token limits work better with small models
      // - More lenient timeouts for slower local inference
      // - Verbose mode helpful for debugging local model behavior
      return {
        ...DEFAULT_CONFIG,
        snapshot: {
          enabled: true,
          limitBase: 60,
          limitStep: 30,
          limitMax: 200,
        },
        retry: {
          verifyTimeoutMs: 15000,
          verifyPollMs: 500,
          verifyMaxAttempts: 6,
          executorRepairAttempts: 3,
          maxReplans: 2,
        },
        plannerMaxTokens: 1024,
        executorMaxTokens: 64,
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
          verifyTimeoutMs: 10000,
          verifyPollMs: 500,
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
          verifyTimeoutMs: 5000,
          verifyPollMs: 500,
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
          verifyTimeoutMs: 20000,
          verifyPollMs: 500,
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
  return {
    ...DEFAULT_CONFIG,
    ...partial,
    snapshot: { ...DEFAULT_CONFIG.snapshot, ...(partial.snapshot ?? {}) },
    retry: { ...DEFAULT_CONFIG.retry, ...(partial.retry ?? {}) },
    stepwise: { ...DEFAULT_CONFIG.stepwise, ...(partial.stepwise ?? {}) },
  };
}
