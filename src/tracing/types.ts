/**
 * Tracing Types
 *
 * Schema v1 - Compatible with Python SDK
 */

/**
 * TraceStats represents execution statistics extracted from a trace
 */
export interface TraceStats {
  total_steps: number;
  total_events: number;
  duration_ms: number | null;
  final_status: 'success' | 'failure' | 'partial' | 'unknown';
  started_at: string | null;
  ended_at: string | null;
}

/**
 * Visual cues structure (matches Element.visual_cues)
 */
export interface TraceVisualCues {
  is_primary: boolean;
  background_color_name: string | null;
  is_clickable: boolean;
}

/**
 * Element data structure for snapshot events
 */
export interface TraceElement {
  id: number;
  bbox: { x: number; y: number; width: number; height: number };
  role: string;
  text?: string | null;
  importance?: number;
  importance_score?: number;
  visual_cues?: TraceVisualCues;
  in_viewport?: boolean;
  is_occluded?: boolean;
  z_index?: number;
  fused_rank_index?: number;
  heuristic_index?: number;
  ml_probability?: number;
  ml_score?: number;
  diff_status?: 'ADDED' | 'REMOVED' | 'MODIFIED' | 'MOVED';
}

/**
 * Pre/post snapshot info for step_end events
 */
export interface SnapshotInfo {
  url?: string;
  snapshot_digest?: string;
  elements?: TraceElement[]; // Include elements with diff_status for diff overlay support
}

/**
 * LLM usage data for step_end events
 */
export interface LLMUsageData {
  model?: string;
  prompt_tokens?: number;
  completion_tokens?: number;
  total_tokens?: number;
  response_text?: string;
  response_hash?: string;
  usage?: {
    prompt_tokens: number;
    completion_tokens: number;
    total_tokens: number;
  };
}

/**
 * Execution data for step_end events
 */
export interface ExecutionData {
  success: boolean;
  action?: string;
  outcome?: string;
  duration_ms?: number;
  element_id?: number;
  bounding_box?: { x: number; y: number; width: number; height: number };
  text?: string;
  key?: string;
  error?: string;
  /** Optional action metadata (e.g., human-like cursor movement path) */
  cursor?: Record<string, any>;
}

/**
 * Element found info for verify signals
 */
export interface ElementFound {
  label: string;
  bounding_box: { x: number; y: number; width: number; height: number };
}

/**
 * Assertion result for verification events
 */
export interface AssertionResult {
  label: string;
  passed: boolean;
  required?: boolean;
  reason?: string;
  details?: Record<string, unknown>;
}

/**
 * Verify signals for step_end events
 */
export interface VerifySignals {
  url_changed?: boolean;
  error?: string;
  elements_found?: ElementFound[];
  // Assertion results from agent verification loop
  assertions?: AssertionResult[];
  task_done?: boolean;
  task_done_label?: string;
}

/**
 * Verify data for step_end events
 */
export interface VerifyData {
  passed: boolean;
  signals: VerifySignals;
}

/**
 * TraceEventData contains fields for event payloads
 * All fields are optional since different event types use different subsets
 */
export interface TraceEventData {
  // Common fields
  goal?: string;
  step_index?: number;
  attempt?: number;
  step_id?: string;
  details?: Record<string, unknown>;

  // Snapshot data
  url?: string;
  element_count?: number;
  timestamp?: string;
  elements?: TraceElement[];
  screenshot_base64?: string;
  screenshot_format?: string;

  // LLM response data
  model?: string;
  prompt_tokens?: number;
  completion_tokens?: number;
  total_tokens?: number;
  response_text?: string;

  // Action data (for action events)
  action_type?: string;
  action?: string; // For step_end events (legacy compatibility)
  element_id?: number;
  text?: string;
  key?: string;
  success?: boolean;
  /** Optional action metadata (e.g., human-like cursor movement path) */
  cursor?: Record<string, any>;

  // Error data
  error?: string;

  // Run metadata
  agent?: string;
  llm_model?: string;
  config?: Record<string, unknown>;
  steps?: number;
  status?: 'success' | 'failure' | 'partial' | 'unknown';

  // Step_end event structure
  v?: number;
  pre?: SnapshotInfo;
  llm?: LLMUsageData;
  exec?: ExecutionData;
  post?: SnapshotInfo;
  verify?: VerifyData;

  // Verification event fields (for assertion loop)
  kind?: 'assert' | 'task_done' | 'captcha' | 'scroll';
  label?: string;
  passed?: boolean;
  required?: boolean;
  reason?: string;
}

/**
 * TraceEvent represents a single event in an agent execution trace
 */
export interface TraceEvent {
  /** Schema version (always 1 for now) */
  v: number;

  /** Event type (e.g., 'run_start', 'snapshot', 'action') */
  type: string;

  /** ISO 8601 timestamp */
  ts: string;

  /** Run UUID */
  run_id: string;

  /** Sequence number (monotonically increasing) */
  seq: number;

  /** Event-specific payload */
  data: TraceEventData;

  /** Optional step UUID (for step-scoped events) */
  step_id?: string;

  /** Optional Unix timestamp in milliseconds */
  ts_ms?: number;
}
