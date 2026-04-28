/**
 * Browser-safe planner-executor entrypoint.
 *
 * This subpath intentionally excludes Playwright, Node filesystem tracing sinks,
 * and the package root's browser automation helpers so Chrome extensions can
 * bundle the planner-executor core behind their own AgentRuntime adapter.
 */

export { LLMProvider, type LLMResponse } from './llm-provider';

export {
  type SnapshotEscalationConfig,
  type RetryConfig,
  type StepwisePlanningConfig,
  type PlannerExecutorConfig,
  type DeepPartial,
  ConfigPreset,
  getConfigPreset,
  mergeConfig,
  DEFAULT_CONFIG,
} from './agents/planner-executor/config';

export {
  PredicateSpecSchema,
  PlanStepSchema,
  PlanSchema,
  ReplanPatchSchema,
  ActionType,
  StepStatus,
  type PredicateSpec,
  type PlanStep,
  type Plan,
  type ReplanPatch,
  type ActionRecord,
  type StepOutcome,
  type RunOutcome,
  type TokenUsageTotals,
  type TokenUsageSummary,
  type SnapshotContext,
  type ParsedAction,
  type Snapshot,
  type SnapshotElement,
} from './agents/planner-executor/plan-models';

export {
  buildStepwisePlannerPrompt,
  buildExecutorPrompt,
  type StepwisePlannerResponse,
} from './agents/planner-executor/prompts';

export {
  parseAction,
  extractJson,
  normalizePlan,
  validatePlanSmoothness,
  formatContext,
} from './agents/planner-executor/plan-utils';

export { TaskCategory, normalizeTaskCategory } from './agents/planner-executor/task-category';
export {
  AutomationTaskSchema,
  type AutomationTask,
} from './agents/planner-executor/automation-task';
export { HeuristicHint, type HeuristicHintInput } from './agents/planner-executor/heuristic-hint';
export { COMMON_HINTS, getCommonHint } from './agents/planner-executor/common-hints';
export {
  ComposableHeuristics,
  type ComposableHeuristicsOptions,
} from './agents/planner-executor/composable-heuristics';

export {
  type Predicate,
  urlContains,
  urlMatches,
  exists,
  notExists,
  elementCount,
  anyOf,
  allOf,
  buildPredicate,
  evaluatePredicates,
} from './agents/planner-executor/predicates';

export {
  PlannerExecutorAgent,
  type PlannerExecutorAgentOptions,
  type PreActionAuthorizer,
  type AuthorizationResult,
  type AgentRuntime,
  type IntentHeuristics,
} from './agents/planner-executor/planner-executor-agent';
