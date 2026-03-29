/**
 * Agents Module
 *
 * High-level agent implementations for browser automation.
 */

// Browser Agent (enterprise features)
export {
  PredicateBrowserAgent,
  type PredicateBrowserAgentConfig,
  type PermissionRecoveryConfig,
  type VisionFallbackConfig,
  type CaptchaConfig,
} from './browser-agent';

// Planner-Executor Agent (two-tier LLM architecture)
export {
  // Configuration
  type SnapshotEscalationConfig,
  type RetryConfig,
  type StepwisePlanningConfig,
  type PlannerExecutorConfig,
  ConfigPreset,
  getConfigPreset,
  mergeConfig,
  DEFAULT_CONFIG,
  // Factory
  type CreateAgentOptions,
  type AgentProviders,
  detectProvider,
  createProvider,
  resolveConfig,
  createPlannerExecutorAgentProviders,
} from './planner-executor';
