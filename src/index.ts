/**
 * Sentience TypeScript SDK - AI Agent Browser Automation
 */

export { SentienceBrowser, PredicateBrowser, PermissionPolicy } from './browser';
export { snapshot, SnapshotOptions, SnapshotGatewayError } from './snapshot';
export { query, find, parseSelector } from './query';
export {
  back,
  check,
  clear,
  click,
  clickRect,
  ClickRect,
  press,
  search,
  scrollTo,
  selectOption,
  sendKeys,
  submit,
  typeText,
  uncheck,
  uploadFile,
} from './actions';
export { CursorPolicy, CursorMode, CursorMovementMetadata, CursorPathPoint } from './cursor-policy';
export { waitFor } from './wait';
export { expect, Expectation } from './expect';
export { Inspector, inspect } from './inspector';
export { Recorder, Trace, TraceStep, record } from './recorder';
export { ScriptGenerator, generate } from './generator';
export { read, extract, ReadOptions, ReadResult } from './read';
export { screenshot, ScreenshotOptions } from './screenshot';
export { showOverlay, clearOverlay } from './overlay';
export { findTextRect } from './textSearch';
export * from './types';
export { saveStorageState } from './utils';
export { getGridBounds } from './utils/grid-utils';

// Agent Layer (v0.2.0+)
export {
  LLMProvider,
  LLMResponse,
  LocalLLMProvider,
  LocalVisionLLMProvider,
  OllamaProvider,
  OpenAIProvider,
  AnthropicProvider,
  GLMProvider,
} from './llm-provider';
export { SentienceAgent, PredicateAgent, AgentActResult, HistoryEntry, TokenStats } from './agent';
export { SentienceVisualAgent, PredicateVisualAgent } from './visual-agent';

// Conversational Agent Layer (v0.3.0+)
export {
  ConversationalAgent,
  ExecutionPlan,
  PlanStep,
  StepResult,
  ConversationEntry,
  ActionType,
  ActionParameters,
} from './conversational-agent';

// Tracing Layer (v0.3.1+)
export {
  Tracer,
  TraceSink,
  JsonlTraceSink,
  CloudTraceSink,
  TraceEvent,
  TraceEventData,
  createTracer,
  createLocalTracer,
  SENTIENCE_API_URL,
} from './tracing';

// Verification Layer (agent assertion loop)
export {
  AssertOutcome,
  AssertContext,
  Predicate,
  downloadCompleted,
  urlMatches,
  urlContains,
  exists,
  notExists,
  elementCount,
  allOf,
  anyOf,
  custom,
  isEnabled,
  isDisabled,
  isChecked,
  isUnchecked,
  valueEquals,
  valueContains,
  isExpanded,
  isCollapsed,
} from './verification';
export { AgentRuntime, AssertionHandle, AssertionRecord, EventuallyOptions } from './agent-runtime';
export { SentienceDebugger, PredicateDebugger } from './debugger';
export { RuntimeAgent } from './runtime-agent';
export type { RuntimeStep, StepVerification } from './runtime-agent';
export { parseVisionExecutorAction, executeVisionExecutorAction } from './vision-executor';
export {
  PredicateBrowserAgent,
  type PredicateBrowserAgentConfig,
  type PermissionRecoveryConfig,
  type VisionFallbackConfig,
  type CaptchaConfig,
  // Planner-Executor Agent (two-tier LLM architecture)
  type SnapshotEscalationConfig,
  type RetryConfig,
  type StepwisePlanningConfig,
  type PlannerExecutorConfig,
  ConfigPreset,
  getConfigPreset,
  mergeConfig,
  DEFAULT_CONFIG,
  type CreateAgentOptions,
  type AgentProviders,
  detectProvider,
  createProvider,
  resolveConfig,
  createPlannerExecutorAgentProviders,
} from './agents';
export * from './captcha/types';
export * from './captcha/strategies';
export * from './tools';

// Ordinal Support (Phase 3)
export {
  OrdinalIntent,
  OrdinalKind,
  detectOrdinalIntent,
  selectByOrdinal,
  boostOrdinalElements,
} from './ordinal';

// Backends (browser-use integration)
export * as backends from './backends';
