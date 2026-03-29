/**
 * Tracer Class
 *
 * High-level API for emitting trace events with automatic sequencing and timestamps
 */

import { TraceSink } from './sink';
import { TraceEvent, TraceEventData } from './types';

/**
 * Tracer provides a high-level API for recording agent execution traces
 */
export class Tracer {
  private runId: string;
  private sink: TraceSink;
  private seq: number;
  private screenshotProcessor?: (screenshot: string) => string;

  // Stats tracking
  private totalSteps: number = 0;
  private totalEvents: number = 0;
  private startedAt: Date | null = null;
  private endedAt: Date | null = null;
  private finalStatus: string = 'unknown';
  // Track step outcomes for automatic status inference
  private stepSuccesses: number = 0;
  private stepFailures: number = 0;
  private hasErrors: boolean = false;

  // Cleanup tracking (for process exit safety net)
  private _closed: boolean = false;
  // Callback invoked when close() is called (used by tracer-factory for cleanup registry)
  public _onCloseCallback?: (tracer: Tracer) => void;

  /**
   * Create a new Tracer
   * @param runId - Unique run identifier (UUID)
   * @param sink - TraceSink implementation (e.g., JsonlTraceSink)
   * @param screenshotProcessor - Optional function to process screenshots before emission.
   *                             Takes base64 string, returns processed base64 string.
   *                             Useful for PII redaction or custom image processing.
   *
   * @example
   * // Basic usage
   * const sink = new JsonlTraceSink('trace.jsonl');
   * const tracer = new Tracer('run-123', sink);
   *
   * @example
   * // With screenshot processor for PII redaction
   * const redactPII = (screenshot: string): string => {
   *   // Your custom redaction logic
   *   return redactedScreenshot;
   * };
   * const tracer = new Tracer('run-123', sink, redactPII);
   */
  constructor(
    runId: string,
    sink: TraceSink,
    screenshotProcessor?: (screenshot: string) => string
  ) {
    this.runId = runId;
    this.sink = sink;
    this.seq = 0;
    this.screenshotProcessor = screenshotProcessor;
  }

  /**
   * Emit a trace event
   * @param eventType - Type of event (e.g., 'run_start', 'snapshot')
   * @param data - Event-specific payload
   * @param stepId - Optional step UUID
   */
  emit(eventType: string, data: TraceEventData, stepId?: string): void {
    this.seq += 1;
    this.totalEvents += 1;

    // Apply screenshot processor if configured and screenshot is present
    if (this.screenshotProcessor && data.screenshot_base64) {
      const processedScreenshot = this.screenshotProcessor(data.screenshot_base64);
      data = { ...data, screenshot_base64: processedScreenshot }; // Don't modify the original object
    }

    // Generate timestamps
    const tsMs = Date.now();
    const ts = new Date(tsMs).toISOString();

    const event: TraceEvent = {
      v: 1,
      type: eventType,
      ts,
      ts_ms: tsMs,
      run_id: this.runId,
      seq: this.seq,
      data,
    };

    if (stepId) {
      event.step_id = stepId;
    }

    this.sink.emit(event);

    // Track step outcomes for automatic status inference
    if (eventType === 'step_end') {
      const success = (data as any).success || false;
      if (success) {
        this.stepSuccesses += 1;
      } else {
        this.stepFailures += 1;
      }
    } else if (eventType === 'error') {
      this.hasErrors = true;
    }
  }

  /**
   * Emit run_start event
   * @param agent - Agent type (e.g., 'SentienceAgent')
   * @param llmModel - Optional LLM model name
   * @param config - Optional configuration
   */
  emitRunStart(agent: string, llmModel?: string, config?: Record<string, any>): void {
    // Track start time
    this.startedAt = new Date();

    const data: TraceEventData = { agent };
    if (llmModel) data.llm_model = llmModel;
    if (config) data.config = config;

    this.emit('run_start', data);
  }

  /**
   * Emit step_start event
   * @param stepId - Step UUID
   * @param stepIndex - Step number (1-indexed)
   * @param goal - Goal description
   * @param attempt - Retry attempt number (0 = first try)
   * @param preUrl - Optional URL before step execution
   */
  emitStepStart(
    stepId: string,
    stepIndex: number,
    goal: string,
    attempt: number = 0,
    preUrl?: string
  ): void {
    // Track step count (only count first attempt of each step)
    if (attempt === 0) {
      this.totalSteps = Math.max(this.totalSteps, stepIndex);
    }

    const data: TraceEventData = {
      step_id: stepId,
      step_index: stepIndex,
      goal,
      attempt,
    };

    if (preUrl) {
      data.url = preUrl;
    }

    this.emit('step_start', data, stepId);
  }

  /**
   * Emit run_end event
   * @param steps - Total number of steps executed
   * @param status - Optional final status ("success", "failure", "partial", "unknown")
   *                 If not provided, infers from tracked outcomes or uses this.finalStatus
   */
  emitRunEnd(steps: number, status?: string): void {
    // Track end time
    this.endedAt = new Date();

    // Auto-infer status if not provided and not explicitly set
    if (status === undefined && this.finalStatus === 'unknown') {
      this._inferFinalStatus();
    }

    // Use provided status or fallback to this.finalStatus
    const finalStatus = status || this.finalStatus;

    // Ensure totalSteps is at least the provided steps value
    this.totalSteps = Math.max(this.totalSteps, steps);

    // Ensure finalStatus is a valid status value
    const validStatus: 'success' | 'failure' | 'partial' | 'unknown' =
      finalStatus === 'success' ||
      finalStatus === 'failure' ||
      finalStatus === 'partial' ||
      finalStatus === 'unknown'
        ? finalStatus
        : 'unknown';

    this.emit('run_end', { steps, status: validStatus });
  }

  /**
   * Emit error event
   * @param stepId - Step UUID where error occurred
   * @param error - Error message
   * @param attempt - Retry attempt number
   */
  emitError(stepId: string, error: string, attempt: number = 0): void {
    this.emit('error', { step_id: stepId, error, attempt }, stepId);
  }

  /**
   * Emit snapshot event with screenshot for Studio visualization.
   *
   * This method builds and emits a 'snapshot' trace event that includes:
   * - Page URL and element data
   * - Screenshot (if present in snapshot)
   * - Step correlation info
   *
   * Use this when you want screenshots to appear in the Sentience Studio timeline.
   *
   * @param snapshot - Snapshot object (must have 'screenshot' property for images)
   * @param stepId - Step UUID (for correlating snapshot with a step)
   * @param stepIndex - Step index (0-based) for Studio timeline ordering
   * @param screenshotFormat - Format of screenshot ("jpeg" or "png", default: "jpeg")
   *
   * @example
   * // After taking a snapshot with AgentRuntime
   * const snapshot = await runtime.snapshot();
   * tracer.emitSnapshot(snapshot, runtime.getStepId(), runtime.getStepIndex());
   *
   * // Or use auto-emit (default in AgentRuntime.snapshot())
   * const snapshot = await runtime.snapshot();  // Auto-emits snapshot event
   */
  emitSnapshot(
    snapshot: any,
    stepId?: string,
    stepIndex?: number,
    screenshotFormat: string = 'jpeg'
  ): void {
    if (!snapshot) {
      return;
    }

    try {
      // Build the snapshot event data
      const data: TraceEventData = {
        url: snapshot.url,
        element_count: snapshot.elements?.length || 0,
        timestamp: snapshot.timestamp,
      };

      // Include step_index if provided (required for UUID step_ids)
      if (stepIndex !== undefined) {
        data.step_index = stepIndex;
      }

      // Include elements data (simplified for trace)
      if (snapshot.elements && snapshot.elements.length > 0) {
        // Normalize importance values to importance_score (0-1 range)
        const importanceValues = snapshot.elements.map((el: any) => el.importance || 0);
        const minImportance = Math.min(...importanceValues);
        const maxImportance = Math.max(...importanceValues);
        const importanceRange = maxImportance - minImportance;

        data.elements = snapshot.elements.map((el: any) => {
          const importanceScore =
            importanceRange > 0 ? (el.importance - minImportance) / importanceRange : 0.5;
          return {
            ...el,
            importance_score: importanceScore,
          };
        });
      }

      // Extract and add screenshot if present
      const screenshotRaw = snapshot.screenshot;
      if (screenshotRaw) {
        // Extract base64 string from data URL if needed
        // Format: "data:image/jpeg;base64,{base64_string}"
        let screenshotBase64: string;
        if (typeof screenshotRaw === 'string' && screenshotRaw.startsWith('data:image')) {
          const commaIndex = screenshotRaw.indexOf(',');
          screenshotBase64 =
            commaIndex !== -1 ? screenshotRaw.slice(commaIndex + 1) : screenshotRaw;
        } else {
          screenshotBase64 = screenshotRaw;
        }
        data.screenshot_base64 = screenshotBase64;
        data.screenshot_format = screenshotFormat;
      }

      this.emit('snapshot', data, stepId);
    } catch {
      // Best-effort: don't let trace emission errors break the caller
    }
  }

  /**
   * Automatically infer finalStatus from tracked step outcomes if not explicitly set.
   * This is called automatically in close() if finalStatus is still "unknown".
   */
  private _inferFinalStatus(): void {
    if (this.finalStatus !== 'unknown') {
      // Status already set explicitly, don't override
      return;
    }

    // Infer from tracked outcomes
    if (this.hasErrors) {
      // Has errors - check if there were successful steps too
      if (this.stepSuccesses > 0) {
        this.finalStatus = 'partial';
      } else {
        this.finalStatus = 'failure';
      }
    } else if (this.stepSuccesses > 0) {
      // Has successful steps and no errors
      this.finalStatus = 'success';
    }
    // Otherwise stays "unknown" (no steps executed or no clear outcome)
  }

  /**
   * Close the underlying sink (flush buffered data)
   * @param blocking - If false, upload happens in background (default: true). Only applies to CloudTraceSink.
   */
  async close(blocking: boolean = true): Promise<void> {
    // Prevent double-close
    if (this._closed) {
      return;
    }
    this._closed = true;

    // Notify cleanup registry (unregister from process exit handler)
    if (this._onCloseCallback) {
      try {
        this._onCloseCallback(this);
      } catch {
        // Don't let callback errors prevent close
      }
    }

    // Auto-infer finalStatus if not explicitly set and we have step outcomes
    if (
      this.finalStatus === 'unknown' &&
      (this.stepSuccesses > 0 || this.stepFailures > 0 || this.hasErrors)
    ) {
      this._inferFinalStatus();
    }

    // Check if sink has a close method that accepts blocking parameter
    if (typeof (this.sink as any).close === 'function' && (this.sink as any).close.length > 0) {
      await (this.sink as any).close(blocking);
    } else {
      await this.sink.close();
    }
  }

  /**
   * Check if tracer has been closed
   */
  isClosed(): boolean {
    return this._closed;
  }

  /**
   * Get run ID
   */
  getRunId(): string {
    return this.runId;
  }

  /**
   * Get current sequence number
   */
  getSeq(): number {
    return this.seq;
  }

  /**
   * Get sink type (for debugging)
   */
  getSinkType(): string {
    return this.sink.getSinkType();
  }

  /**
   * Set the final status of the trace run
   * @param status - Final status ("success", "failure", "partial", "unknown")
   */
  setFinalStatus(status: string): void {
    if (!['success', 'failure', 'partial', 'unknown'].includes(status)) {
      throw new Error(
        `Invalid status: ${status}. Must be one of: success, failure, partial, unknown`
      );
    }
    this.finalStatus = status;
  }

  /**
   * Get execution statistics for trace completion
   * @returns Dictionary with stats fields for /v1/traces/complete
   */
  getStats(): Record<string, any> {
    let durationMs: number | null = null;
    if (this.startedAt && this.endedAt) {
      durationMs = this.endedAt.getTime() - this.startedAt.getTime();
    }

    return {
      total_steps: this.totalSteps,
      total_events: this.totalEvents,
      duration_ms: durationMs,
      final_status: this.finalStatus,
      started_at: this.startedAt ? this.startedAt.toISOString() : null,
      ended_at: this.endedAt ? this.endedAt.toISOString() : null,
    };
  }
}
