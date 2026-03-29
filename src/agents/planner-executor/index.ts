/**
 * PlannerExecutorAgent Module
 *
 * Two-tier LLM architecture for browser automation:
 * - Planner (7B+ model): Generates JSON execution plans
 * - Executor (3B-7B model): Executes steps with tight prompts
 *
 * Note: The full PlannerExecutorAgent class is not yet ported to TypeScript.
 * This module provides configuration and factory helpers for when it is.
 */

// Configuration
export {
  SnapshotEscalationConfig,
  RetryConfig,
  StepwisePlanningConfig,
  PlannerExecutorConfig,
  DeepPartial,
  ConfigPreset,
  getConfigPreset,
  mergeConfig,
  DEFAULT_CONFIG,
} from './config';

// Factory
export {
  CreateAgentOptions,
  AgentProviders,
  detectProvider,
  createProvider,
  resolveConfig,
  createPlannerExecutorAgentProviders,
} from './agent-factory';
