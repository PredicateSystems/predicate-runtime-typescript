/**
 * JSONL Trace Sink
 *
 * Writes trace events to a local JSONL (JSON Lines) file
 */

import * as fs from 'fs';
import * as path from 'path';
import { TraceSink } from './sink';
import { TraceEvent, TraceStats } from './types';
import { writeTraceIndex } from './indexer';

/**
 * JsonlTraceSink writes trace events to a JSONL file (one JSON object per line)
 */
export class JsonlTraceSink extends TraceSink {
  private path: string;
  private writeStream: fs.WriteStream | null = null;
  private closed: boolean = false;

  /**
   * Create a new JSONL trace sink
   * @param filePath - Path to the JSONL file (will be created if doesn't exist)
   */
  constructor(filePath: string) {
    super();
    this.path = filePath;

    // Create parent directories if needed (synchronously)
    const dir = path.dirname(filePath);
    try {
      if (!fs.existsSync(dir)) {
        fs.mkdirSync(dir, { recursive: true });
      }

      // Verify directory is writable
      fs.accessSync(dir, fs.constants.W_OK);

      // Open file in append mode with line buffering
      this.writeStream = fs.createWriteStream(filePath, {
        flags: 'a',
        encoding: 'utf-8',
        autoClose: true,
      });

      // Handle stream errors (suppress logging if stream is closed)
      this.writeStream.on('error', error => {
        if (!this.closed) {
          console.error('[JsonlTraceSink] Stream error:', error);
        }
      });
    } catch (error) {
      console.error('[JsonlTraceSink] Failed to initialize sink:', error);
      this.writeStream = null;
    }
  }

  /**
   * Emit a trace event (write as JSON line)
   * @param event - Trace event to emit
   */
  emit(event: TraceEvent): void {
    if (this.closed) {
      // Only warn in non-test environments to avoid test noise
      const isTestEnv =
        process.env.CI === 'true' ||
        process.env.NODE_ENV === 'test' ||
        process.env.JEST_WORKER_ID !== undefined ||
        (typeof global !== 'undefined' && (global as any).__JEST__);

      if (!isTestEnv) {
        console.warn('[JsonlTraceSink] Attempted to emit after close()');
      }
      return;
    }

    if (!this.writeStream) {
      console.error('[JsonlTraceSink] Write stream not available');
      return;
    }

    try {
      const jsonLine = JSON.stringify(event) + '\n';
      const written = this.writeStream.write(jsonLine);
      // If write returns false, the stream is backpressured
      // We don't need to wait, but we could add a drain listener if needed
      if (!written) {
        // Stream is backpressured - wait for drain
        this.writeStream.once('drain', () => {
          // Stream is ready again
        });
      }
    } catch (error) {
      // Log error but don't crash agent execution
      console.error('[JsonlTraceSink] Failed to write event:', error);
    }
  }

  /**
   * Close the sink and flush buffered data
   */
  async close(): Promise<void> {
    if (this.closed) {
      return;
    }

    this.closed = true;

    // Check if stream exists and is writable
    if (!this.writeStream || this.writeStream.destroyed) {
      return;
    }

    // Store reference to satisfy TypeScript null checks
    const stream = this.writeStream;

    // Remove error listener to prevent late errors
    stream.removeAllListeners('error');

    return new Promise<void>(resolve => {
      // Check if stream is already closed
      if (stream.destroyed || !stream.writable) {
        // Stream already closed, generate index and resolve immediately
        this.generateIndex();
        resolve();
        return;
      }

      let resolved = false;
      const doResolve = () => {
        if (!resolved) {
          resolved = true;
          clearTimeout(timeout);
          // Generate index after closing file
          this.generateIndex();
          resolve();
        }
      };

      // Fallback timeout in case 'close' event doesn't fire (shouldn't happen, but safety)
      const timeout = setTimeout(() => {
        if (!resolved) {
          doResolve();
        }
      }, 500);

      // Wait for stream to fully close (Windows needs this)
      // The 'close' event fires after all data is flushed and file handle is released
      stream.once('close', doResolve);

      stream.end((err?: Error | null) => {
        if (err) {
          // Silently ignore close errors in production
          // (they're logged during stream lifetime if needed)
        }
        // Note: 'close' event will fire after end() completes
        // Don't resolve here - wait for 'close' event
      });
    });
  }

  /**
   * Generate trace index file (automatic on close)
   */
  private generateIndex(): void {
    try {
      // Use frontend format to ensure 'step' field is present (1-based)
      // Frontend derives sequence from step.step - 1, so step must be valid
      const indexPath = this.path.replace(/\.jsonl$/, '.index.json');
      writeTraceIndex(this.path, indexPath, true);
    } catch (error: any) {
      // Non-fatal: log but don't crash
      console.log(`⚠️  Failed to generate trace index: ${error.message}`);
    }
  }

  /**
   * Get sink type identifier
   */
  getSinkType(): string {
    return `JsonlTraceSink(${this.path})`;
  }

  /**
   * Get file path
   */
  getPath(): string {
    return this.path;
  }

  /**
   * Check if sink is closed
   */
  isClosed(): boolean {
    return this.closed;
  }

  /**
   * Extract execution statistics from trace file (for local traces).
   * @returns Trace statistics
   */
  getStats(): TraceStats {
    try {
      // Read trace file to extract stats
      const traceContent = fs.readFileSync(this.path, 'utf-8');
      const lines = traceContent.split('\n').filter(line => line.trim());
      const events: TraceEvent[] = [];

      for (const line of lines) {
        try {
          const event = JSON.parse(line);
          events.push(event);
        } catch {
          continue;
        }
      }

      if (events.length === 0) {
        return {
          total_steps: 0,
          total_events: 0,
          duration_ms: null,
          final_status: 'unknown',
          started_at: null,
          ended_at: null,
        };
      }

      // Find run_start and run_end events
      const runStart = events.find(e => e.type === 'run_start');
      const runEnd = events.find(e => e.type === 'run_end');

      // Extract timestamps
      const startedAt = runStart?.ts || null;
      const endedAt = runEnd?.ts || null;

      // Calculate duration
      let durationMs: number | null = null;
      if (startedAt && endedAt) {
        try {
          const startDt = new Date(startedAt);
          const endDt = new Date(endedAt);
          durationMs = endDt.getTime() - startDt.getTime();
        } catch {
          // Ignore parse errors
        }
      }

      // Count steps (from step_start events, only first attempt)
      const stepIndices = new Set<number>();
      for (const event of events) {
        if (event.type === 'step_start') {
          const stepIndex = event.data?.step_index;
          if (stepIndex !== undefined) {
            stepIndices.add(stepIndex);
          }
        }
      }
      let totalSteps = stepIndices.size;

      // If run_end has steps count, use that (more accurate)
      if (runEnd) {
        const stepsFromEnd = runEnd.data?.steps;
        if (stepsFromEnd !== undefined) {
          totalSteps = Math.max(totalSteps, stepsFromEnd);
        }
      }

      // Count total events
      const totalEvents = events.length;

      // Infer final status
      let finalStatus: TraceStats['final_status'] = 'unknown';
      // Check for run_end event with status
      if (runEnd) {
        const status = runEnd.data?.status;
        if (
          status === 'success' ||
          status === 'failure' ||
          status === 'partial' ||
          status === 'unknown'
        ) {
          finalStatus = status;
        }
      } else {
        // Infer from error events
        const hasErrors = events.some(e => e.type === 'error');
        if (hasErrors) {
          const stepEnds = events.filter(e => e.type === 'step_end');
          if (stepEnds.length > 0) {
            finalStatus = 'partial';
          } else {
            finalStatus = 'failure';
          }
        } else {
          const stepEnds = events.filter(e => e.type === 'step_end');
          if (stepEnds.length > 0) {
            finalStatus = 'success';
          }
        }
      }

      return {
        total_steps: totalSteps,
        total_events: totalEvents,
        duration_ms: durationMs,
        final_status: finalStatus,
        started_at: startedAt,
        ended_at: endedAt,
      };
    } catch {
      return {
        total_steps: 0,
        total_events: 0,
        duration_ms: null,
        final_status: 'unknown',
        started_at: null,
        ended_at: null,
      };
    }
  }
}
