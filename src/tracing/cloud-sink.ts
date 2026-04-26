/**
 * CloudTraceSink - Enterprise Cloud Upload
 *
 * Implements "Local Write, Batch Upload" pattern for cloud tracing
 *
 * PRODUCTION HARDENING:
 * - Uses persistent cache directory (~/.sentience/traces/pending/) to survive crashes
 * - Supports non-blocking close() to avoid hanging user scripts
 * - Preserves traces locally on upload failure
 */

import * as fs from 'fs';
import { promises as fsPromises } from 'fs';
import * as os from 'os';
import * as path from 'path';
import * as zlib from 'zlib';
import * as https from 'https';
import * as http from 'http';
import { URL } from 'url';
import { TraceSink } from './sink';
import { TraceEvent, TraceStats } from './types';
import { writeTraceIndex } from './indexer';

/**
 * Optional logger interface for SDK users
 */
export interface SentienceLogger {
  info(message: string): void;
  warn(message: string): void;
  error(message: string): void;
}

/**
 * Get persistent cache directory for traces
 * Uses ~/.sentience/traces/pending/ (survives process crashes)
 */
function getPersistentCacheDir(): string {
  const homeDir = os.homedir();
  const cacheDir = path.join(homeDir, '.sentience', 'traces', 'pending');

  // Create directory if it doesn't exist
  if (!fs.existsSync(cacheDir)) {
    fs.mkdirSync(cacheDir, { recursive: true });
  }

  return cacheDir;
}

/**
 * CloudTraceSink writes trace events to a local temp file,
 * then uploads the complete trace to cloud storage on close()
 *
 * Architecture:
 * 1. **Local Buffer**: Writes to temp file (zero latency, non-blocking)
 * 2. **Pre-signed URL**: Uses secure pre-signed PUT URL from backend API
 * 3. **Batch Upload**: Uploads complete file on close() or at intervals
 * 4. **Zero Credential Exposure**: Never embeds cloud credentials in SDK
 *
 * This design ensures:
 * - Fast agent performance (microseconds per emit, not milliseconds)
 * - Security (credentials stay on backend)
 * - Reliability (network issues don't crash the agent)
 *
 * Example:
 *   const sink = new CloudTraceSink(uploadUrl);
 *   const tracer = new Tracer(runId, sink);
 *   tracer.emitRunStart('SentienceAgent');
 *   await tracer.close(); // Uploads to cloud
 */
export class CloudTraceSink extends TraceSink {
  private uploadUrl: string;
  private tempFilePath: string;
  private runId: string;
  private writeStream: fs.WriteStream | null = null;
  private closed: boolean = false;
  private closePromise: Promise<void> | null = null;
  private suppressCloseLogs: boolean = false;
  private apiKey?: string;
  private apiUrl: string;
  private logger?: SentienceLogger;

  // File size tracking
  private traceFileSizeBytes: number = 0;
  private screenshotTotalSizeBytes: number = 0;
  private screenshotCount: number = 0; // Track number of screenshots extracted
  private indexFileSizeBytes: number = 0; // Track index file size

  // Upload success flag
  private uploadSuccessful: boolean = false;

  /**
   * Create a new CloudTraceSink
   *
   * @param uploadUrl - Pre-signed PUT URL from Sentience API
   * @param runId - Run ID for persistent cache naming
   * @param apiKey - Sentience API key for calling /v1/traces/complete
   * @param apiUrl - Sentience API base URL (default: https://api.sentienceapi.com)
   * @param logger - Optional logger instance for logging file sizes and errors
   */
  constructor(
    uploadUrl: string,
    runId?: string,
    apiKey?: string,
    apiUrl?: string,
    logger?: SentienceLogger
  ) {
    super();
    this.uploadUrl = uploadUrl;
    this.runId = runId || `trace-${Date.now()}`;
    this.apiKey = apiKey;
    this.apiUrl = apiUrl || 'https://api.sentienceapi.com';
    this.logger = logger;

    // PRODUCTION FIX: Use persistent cache directory instead of /tmp
    // This ensures traces survive process crashes!
    const cacheDir = getPersistentCacheDir();
    this.tempFilePath = path.join(cacheDir, `${this.runId}.jsonl`);

    try {
      // Open file in append mode
      this.writeStream = fs.createWriteStream(this.tempFilePath, {
        flags: 'a',
        encoding: 'utf-8',
        autoClose: true,
      });

      // Handle stream errors (suppress if closed)
      this.writeStream.on('error', error => {
        if (!this.closed) {
          console.error('[CloudTraceSink] Stream error:', error);
        }
      });
    } catch (error) {
      console.error('[CloudTraceSink] Failed to initialize sink:', error);
      this.writeStream = null;
    }
  }

  /**
   * Emit a trace event to local temp file (fast, non-blocking)
   *
   * @param event - Trace event to emit
   */
  emit(event: TraceEvent): void {
    if (this.closed) {
      throw new Error('CloudTraceSink is closed');
    }

    if (!this.writeStream) {
      console.error('[CloudTraceSink] Write stream not available');
      return;
    }

    try {
      const jsonStr = JSON.stringify(event);
      this.writeStream.write(jsonStr + '\n');
    } catch (error) {
      console.error('[CloudTraceSink] Write error:', error);
    }
  }

  /**
   * Upload data to cloud using Node's built-in https module
   */
  private async _uploadToCloud(data: Buffer): Promise<number> {
    return new Promise((resolve, reject) => {
      const url = new URL(this.uploadUrl);
      const protocol = url.protocol === 'https:' ? https : http;

      const options = {
        hostname: url.hostname,
        port: url.port || (url.protocol === 'https:' ? 443 : 80),
        path: url.pathname + url.search,
        method: 'PUT',
        headers: {
          'Content-Type': 'application/x-gzip',
          'Content-Encoding': 'gzip',
          'Content-Length': data.length,
        },
        timeout: 60000, // 1 minute timeout
      };

      const req = protocol.request(options, res => {
        // Consume response data (even if we don't use it)
        res.on('data', () => {});
        res.on('end', () => {
          resolve(res.statusCode || 500);
        });
      });

      req.on('error', error => {
        reject(error);
      });

      req.on('timeout', () => {
        req.destroy();
        reject(new Error('Upload timeout'));
      });

      req.write(data);
      req.end();
    });
  }

  /**
   * Upload buffered trace to cloud via pre-signed URL
   *
   * @param blocking - If false, upload happens in background (default: true)
   *
   * PRODUCTION FIX: Non-blocking mode prevents hanging user scripts
   * on slow uploads (Risk #2 from production hardening plan)
   */
  async close(_blocking: boolean = true): Promise<void> {
    if (this.closePromise) {
      return this.closePromise;
    }

    this.closed = true;
    this.suppressCloseLogs = !_blocking && Boolean(process.env.JEST_WORKER_ID);
    this.closePromise = this._doUpload();

    // `blocking=false` remains non-blocking for callers that ignore the returned promise.
    // Callers that do await it now correctly wait for the upload/cleanup path to settle.
    return this.closePromise;
  }

  /**
   * Internal upload logic (called by both blocking and non-blocking close)
   */
  private async _doUpload(): Promise<void> {
    try {
      // 1. Close write stream
      if (this.writeStream && !this.writeStream.destroyed) {
        const stream = this.writeStream;
        stream.removeAllListeners('error');

        await new Promise<void>(resolve => {
          stream.end(() => {
            resolve();
          });
        });
      }

      // 2. Generate index after closing file
      this.generateIndex();

      // 2. Check trace file exists
      try {
        await fsPromises.access(this.tempFilePath);
      } catch {
        this.warnDuringClose('[CloudTraceSink] Temp file does not exist, skipping upload');
        return;
      }

      // 3. Extract screenshots from trace events
      const screenshots = await this._extractScreenshotsFromTrace();
      this.screenshotCount = screenshots.size;

      // 4. Upload screenshots separately
      if (screenshots.size > 0) {
        await this._uploadScreenshots(screenshots);
      }

      // 5. Create cleaned trace file (without screenshot_base64)
      const cleanedTracePath = this.tempFilePath.replace('.jsonl', '.cleaned.jsonl');
      await this._createCleanedTrace(cleanedTracePath);

      // 6. Read and compress cleaned trace
      const traceData = await fsPromises.readFile(cleanedTracePath);
      const compressedData = zlib.gzipSync(traceData);

      // Measure trace file size
      this.traceFileSizeBytes = compressedData.length;

      // Log file sizes if logger is provided
      if (this.logger) {
        this.logger.info(
          `Trace file size: ${(this.traceFileSizeBytes / 1024 / 1024).toFixed(2)} MB`
        );
        this.logger.info(
          `Screenshot total: ${(this.screenshotTotalSizeBytes / 1024 / 1024).toFixed(2)} MB`
        );
      }

      // 7. Upload cleaned trace to cloud
      if (this.logger) {
        this.logger.info(`Uploading trace to cloud (${compressedData.length} bytes)`);
      }

      const statusCode = await this._uploadToCloud(compressedData);

      if (statusCode === 200) {
        this.uploadSuccessful = true;
        if (this.logger) {
          this.logger.info('Trace uploaded successfully');
        }

        // Upload trace index file
        await this._uploadIndex();

        // Call /v1/traces/complete to report file sizes
        await this._completeTrace();

        // 8. Delete files on success
        await this._cleanupFiles();

        // Clean up temporary cleaned trace file
        try {
          await fsPromises.unlink(cleanedTracePath);
        } catch {
          // Ignore cleanup errors
        }
      } else {
        this.uploadSuccessful = false;
        this.errorDuringClose(`❌ [Sentience] Upload failed: HTTP ${statusCode}`);
        this.errorDuringClose(`   Local trace preserved at: ${this.tempFilePath}`);
      }
    } catch (error: any) {
      this.errorDuringClose(`❌ [Sentience] Error uploading trace: ${error.message}`);
      this.errorDuringClose(`   Local trace preserved at: ${this.tempFilePath}`);
      // Don't throw - preserve trace locally even if upload fails
    }
  }

  private warnDuringClose(message: string): void {
    if (this.logger) {
      this.logger.warn(message);
      return;
    }

    if (!this.suppressCloseLogs) {
      console.warn(message);
    }
  }

  private errorDuringClose(message: string): void {
    if (this.logger) {
      this.logger.error(message);
      return;
    }

    if (!this.suppressCloseLogs) {
      console.error(message);
    }
  }

  /**
   * Infer final status from trace events by reading the trace file.
   * @returns Final status: "success", "failure", "partial", or "unknown"
   */
  private _inferFinalStatusFromTrace(): string {
    try {
      // Read trace file to analyze events
      const traceContent = fs.readFileSync(this.tempFilePath, 'utf-8');
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
        return 'unknown';
      }

      // Check for run_end event with status
      for (let i = events.length - 1; i >= 0; i--) {
        const event = events[i];
        if (event.type === 'run_end') {
          const status = event.data?.status;
          if (
            status === 'success' ||
            status === 'failure' ||
            status === 'partial' ||
            status === 'unknown'
          ) {
            return status;
          }
        }
      }

      // Infer from error events
      const hasErrors = events.some(e => e.type === 'error');
      if (hasErrors) {
        // Check if there are successful steps too (partial success)
        const stepEnds = events.filter(e => e.type === 'step_end');
        if (stepEnds.length > 0) {
          return 'partial';
        }
        return 'failure';
      }

      // If we have step_end events and no errors, likely success
      const stepEnds = events.filter(e => e.type === 'step_end');
      if (stepEnds.length > 0) {
        return 'success';
      }

      return 'unknown';
    } catch {
      // If we can't read the trace, default to unknown
      return 'unknown';
    }
  }

  /**
   * Extract execution statistics from trace file.
   * @returns Trace statistics for /v1/traces/complete
   */
  private _extractStatsFromTrace(): TraceStats {
    try {
      // Read trace file to extract stats
      const traceContent = fs.readFileSync(this.tempFilePath, 'utf-8');
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
      const finalStatus = this._inferFinalStatusFromTrace();

      return {
        total_steps: totalSteps,
        total_events: totalEvents,
        duration_ms: durationMs,
        final_status: finalStatus as TraceStats['final_status'],
        started_at: startedAt,
        ended_at: endedAt,
      };
    } catch (error: any) {
      this.logger?.warn(`Error extracting stats from trace: ${error.message}`);
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

  /**
   * Call /v1/traces/complete to report file sizes and stats to gateway.
   *
   * This is a best-effort call - failures are logged but don't affect upload success.
   */
  private async _completeTrace(): Promise<void> {
    if (!this.apiKey) {
      // No API key - skip complete call
      return;
    }

    return new Promise(resolve => {
      const url = new URL(`${this.apiUrl}/v1/traces/complete`);
      const protocol = url.protocol === 'https:' ? https : http;

      // Extract stats from trace file
      const stats = this._extractStatsFromTrace();

      // Add file size fields
      const completeStats = {
        ...stats,
        trace_file_size_bytes: this.traceFileSizeBytes,
        screenshot_total_size_bytes: this.screenshotTotalSizeBytes,
        screenshot_count: this.screenshotCount,
        index_file_size_bytes: this.indexFileSizeBytes,
      };

      const body = JSON.stringify({
        run_id: this.runId,
        stats: completeStats,
      });

      const options = {
        hostname: url.hostname,
        port: url.port || (url.protocol === 'https:' ? 443 : 80),
        path: url.pathname + url.search,
        method: 'POST',
        headers: {
          'Content-Type': 'application/json',
          'Content-Length': Buffer.byteLength(body),
          Authorization: `Bearer ${this.apiKey}`,
        },
        timeout: 10000, // 10 second timeout
      };

      const req = protocol.request(options, res => {
        // Consume response data
        res.on('data', () => {});
        res.on('end', () => {
          if (res.statusCode === 200) {
            this.logger?.info('Trace completion reported to gateway');
          } else {
            this.logger?.warn(`Failed to report trace completion: HTTP ${res.statusCode}`);
          }
          resolve();
        });
      });

      req.on('error', error => {
        // Best-effort - log but don't fail
        this.logger?.warn(`Error reporting trace completion: ${error.message}`);
        resolve();
      });

      req.on('timeout', () => {
        req.destroy();
        this.logger?.warn('Trace completion request timeout');
        resolve();
      });

      req.write(body);
      req.end();
    });
  }

  /**
   * Generate trace index file (automatic on close)
   */
  private generateIndex(): void {
    try {
      // Use frontend format to ensure 'step' field is present (1-based)
      // Frontend derives sequence from step.step - 1, so step must be valid
      const indexPath = this.tempFilePath.replace('.jsonl', '.index.json');
      writeTraceIndex(this.tempFilePath, indexPath, true);
    } catch (error: any) {
      // Non-fatal: log but don't crash
      this.logger?.warn(`Failed to generate trace index: ${error.message}`);
    }
  }

  /**
   * Upload trace index file to cloud storage.
   *
   * Called after successful trace upload to provide fast timeline rendering.
   * The index file enables O(1) step lookups without parsing the entire trace.
   */
  private async _uploadIndex(): Promise<void> {
    // Construct index file path (same as trace file with .index.json extension)
    const indexPath = this.tempFilePath.replace('.jsonl', '.index.json');

    try {
      // Check if index file exists
      await fsPromises.access(indexPath);
    } catch {
      this.logger?.warn('Index file not found, skipping index upload');
      return;
    }

    try {
      // Request index upload URL from API
      if (!this.apiKey) {
        this.logger?.info('No API key provided, skipping index upload');
        return;
      }

      const uploadUrlResponse = await this._requestIndexUploadUrl();
      if (!uploadUrlResponse) {
        return;
      }

      // Read index file and update trace_file.path to cloud storage path
      const indexContent = await fsPromises.readFile(indexPath, 'utf-8');
      const indexJson = JSON.parse(indexContent);

      // Extract cloud storage path from trace upload URL
      // uploadUrl format: https://...digitaloceanspaces.com/traces/{run_id}.jsonl.gz
      // Extract path: traces/{run_id}.jsonl.gz
      try {
        const parsedUrl = new URL(this.uploadUrl);
        // Extract path after domain (e.g., /traces/run-123.jsonl.gz -> traces/run-123.jsonl.gz)
        const cloudTracePath = parsedUrl.pathname.startsWith('/')
          ? parsedUrl.pathname.substring(1)
          : parsedUrl.pathname;
        // Update trace_file.path in index
        if (indexJson.trace_file && typeof indexJson.trace_file === 'object') {
          indexJson.trace_file.path = cloudTracePath;
        }
      } catch (error: any) {
        this.logger?.warn(`Failed to extract cloud path from upload URL: ${error.message}`);
      }

      // Serialize updated index to JSON
      const updatedIndexData = Buffer.from(JSON.stringify(indexJson, null, 2), 'utf-8');
      const compressedIndex = zlib.gzipSync(updatedIndexData);
      const indexSize = compressedIndex.length;
      this.indexFileSizeBytes = indexSize; // Track index file size

      this.logger?.info(`Index file size: ${(indexSize / 1024).toFixed(2)} KB`);
      if (this.logger) {
        this.logger.info(`Uploading trace index (${indexSize} bytes)`);
      }

      // Upload index to cloud storage
      const statusCode = await this._uploadIndexToCloud(uploadUrlResponse, compressedIndex);

      if (statusCode === 200) {
        if (this.logger) {
          this.logger.info('Trace index uploaded successfully');
        }

        // Delete local index file after successful upload
        try {
          await fsPromises.unlink(indexPath);
        } catch {
          // Ignore cleanup errors
        }
      } else {
        this.logger?.warn(`Index upload failed: HTTP ${statusCode}`);
      }
    } catch (error: any) {
      // Non-fatal: log but don't crash
      this.logger?.warn(`Error uploading trace index: ${error.message}`);
    }
  }

  /**
   * Request index upload URL from Sentience API
   */
  private async _requestIndexUploadUrl(): Promise<string | null> {
    return new Promise(resolve => {
      const url = new URL(`${this.apiUrl}/v1/traces/index_upload`);
      const protocol = url.protocol === 'https:' ? https : http;

      const body = JSON.stringify({ run_id: this.runId });

      const options = {
        hostname: url.hostname,
        port: url.port || (url.protocol === 'https:' ? 443 : 80),
        path: url.pathname + url.search,
        method: 'POST',
        headers: {
          'Content-Type': 'application/json',
          'Content-Length': Buffer.byteLength(body),
          Authorization: `Bearer ${this.apiKey}`,
        },
        timeout: 10000,
      };

      const req = protocol.request(options, res => {
        let data = '';
        res.on('data', chunk => {
          data += chunk;
        });
        res.on('end', () => {
          if (res.statusCode === 200) {
            try {
              const response = JSON.parse(data);
              resolve(response.upload_url || null);
            } catch {
              this.logger?.warn('Failed to parse index upload URL response');
              resolve(null);
            }
          } else {
            this.logger?.warn(`Failed to get index upload URL: HTTP ${res.statusCode}`);
            resolve(null);
          }
        });
      });

      req.on('error', error => {
        this.logger?.warn(`Error requesting index upload URL: ${error.message}`);
        resolve(null);
      });

      req.on('timeout', () => {
        req.destroy();
        this.logger?.warn('Index upload URL request timeout');
        resolve(null);
      });

      req.write(body);
      req.end();
    });
  }

  /**
   * Upload index data to cloud using pre-signed URL
   */
  private async _uploadIndexToCloud(uploadUrl: string, data: Buffer): Promise<number> {
    return new Promise((resolve, reject) => {
      const url = new URL(uploadUrl);
      const protocol = url.protocol === 'https:' ? https : http;

      const options = {
        hostname: url.hostname,
        port: url.port || (url.protocol === 'https:' ? 443 : 80),
        path: url.pathname + url.search,
        method: 'PUT',
        headers: {
          'Content-Type': 'application/json',
          'Content-Encoding': 'gzip',
          'Content-Length': data.length,
        },
        timeout: 30000, // 30 second timeout
      };

      const req = protocol.request(options, res => {
        res.on('data', () => {});
        res.on('end', () => {
          resolve(res.statusCode || 500);
        });
      });

      req.on('error', error => {
        reject(error);
      });

      req.on('timeout', () => {
        req.destroy();
        reject(new Error('Index upload timeout'));
      });

      req.write(data);
      req.end();
    });
  }

  /**
   * Normalize screenshot data by extracting base64 from data URL if needed.
   *
   * Handles both formats:
   * - Data URL: "data:image/jpeg;base64,/9j/4AAQ..."
   * - Pure base64: "/9j/4AAQ..."
   *
   * @param screenshotRaw - Raw screenshot data (data URL or base64)
   * @param defaultFormat - Default format if not detected from data URL
   * @returns Tuple of [base64String, formatString]
   */
  private _normalizeScreenshotData(
    screenshotRaw: string,
    defaultFormat: string = 'jpeg'
  ): [string, string] {
    if (!screenshotRaw) {
      return ['', defaultFormat];
    }

    // Check if it's a data URL
    if (screenshotRaw.startsWith('data:image')) {
      // Extract format from "data:image/jpeg;base64,..." or "data:image/png;base64,..."
      try {
        // Split on comma to get the base64 part
        if (screenshotRaw.includes(',')) {
          const [header, base64Data] = screenshotRaw.split(',', 2);
          // Extract format from header: "data:image/jpeg;base64"
          if (header.includes('/') && header.includes(';')) {
            const formatPart = header.split('/')[1]?.split(';')[0];
            if (formatPart === 'jpeg' || formatPart === 'jpg') {
              return [base64Data, 'jpeg'];
            } else if (formatPart === 'png') {
              return [base64Data, 'png'];
            }
          }
          return [base64Data, defaultFormat];
        } else {
          // Malformed data URL - return as-is with warning
          this.logger?.warn('Malformed data URL in screenshot_base64 (missing comma)');
          return [screenshotRaw, defaultFormat];
        }
      } catch (error: any) {
        this.logger?.warn(`Error parsing screenshot data URL: ${error.message}`);
        return [screenshotRaw, defaultFormat];
      }
    }

    // Already pure base64
    return [screenshotRaw, defaultFormat];
  }

  /**
   * Extract screenshots from trace events.
   *
   * @returns Map of sequence number to screenshot data
   */
  private async _extractScreenshotsFromTrace(): Promise<
    Map<number, { base64: string; format: string; stepId?: string }>
  > {
    const screenshots = new Map<number, { base64: string; format: string; stepId?: string }>();
    let sequence = 0;

    try {
      const traceContent = await fsPromises.readFile(this.tempFilePath, 'utf-8');
      const lines = traceContent.split('\n');

      for (const line of lines) {
        if (!line.trim()) {
          continue;
        }

        try {
          const event = JSON.parse(line);
          // Check if this is a snapshot event with screenshot
          if (event.type === 'snapshot') {
            const data = event.data || {};
            const screenshotRaw = data.screenshot_base64;

            if (screenshotRaw) {
              // Normalize: extract base64 from data URL if needed
              // Handles both "data:image/jpeg;base64,..." and pure base64
              const [screenshotBase64, screenshotFormat] = this._normalizeScreenshotData(
                screenshotRaw,
                data.screenshot_format || 'jpeg'
              );

              if (screenshotBase64) {
                sequence += 1;
                screenshots.set(sequence, {
                  base64: screenshotBase64,
                  format: screenshotFormat,
                  stepId: event.step_id,
                });
              }
            }
          }
        } catch {
          // Skip invalid JSON lines
          continue;
        }
      }
    } catch (error: any) {
      this.logger?.error(`Error extracting screenshots: ${error.message}`);
    }

    return screenshots;
  }

  /**
   * Create trace file without screenshot_base64 fields.
   *
   * @param outputPath - Path to write cleaned trace file
   */
  private async _createCleanedTrace(outputPath: string): Promise<void> {
    try {
      const traceContent = await fsPromises.readFile(this.tempFilePath, 'utf-8');
      const lines = traceContent.split('\n');
      const cleanedLines: string[] = [];

      for (const line of lines) {
        if (!line.trim()) {
          continue;
        }

        try {
          const event = JSON.parse(line);
          // Remove screenshot_base64 from snapshot events
          if (event.type === 'snapshot' && event.data) {
            const cleanedData: any = {};
            for (const [key, value] of Object.entries(event.data)) {
              if (key !== 'screenshot_base64' && key !== 'screenshot_format') {
                cleanedData[key] = value;
              }
            }
            event.data = cleanedData;
          }

          cleanedLines.push(JSON.stringify(event));
        } catch {
          // Skip invalid JSON lines
          continue;
        }
      }

      await fsPromises.writeFile(outputPath, cleanedLines.join('\n') + '\n', 'utf-8');
    } catch (error: any) {
      this.logger?.error(`Error creating cleaned trace: ${error.message}`);
      throw error;
    }
  }

  /**
   * Request pre-signed upload URLs for screenshots from gateway.
   *
   * @param sequences - List of screenshot sequence numbers
   * @returns Map of sequence number to upload URL
   */
  private async _requestScreenshotUrls(sequences: number[]): Promise<Map<number, string>> {
    if (!this.apiKey || sequences.length === 0) {
      return new Map();
    }

    return new Promise(resolve => {
      const url = new URL(`${this.apiUrl}/v1/screenshots/init`);
      const protocol = url.protocol === 'https:' ? https : http;

      const body = JSON.stringify({
        run_id: this.runId,
        sequences,
      });

      const options = {
        hostname: url.hostname,
        port: url.port || (url.protocol === 'https:' ? 443 : 80),
        path: url.pathname + url.search,
        method: 'POST',
        headers: {
          'Content-Type': 'application/json',
          'Content-Length': Buffer.byteLength(body),
          Authorization: `Bearer ${this.apiKey}`,
        },
        timeout: 10000, // 10 second timeout
      };

      const req = protocol.request(options, res => {
        let data = '';
        res.on('data', chunk => {
          data += chunk;
        });
        res.on('end', () => {
          if (res.statusCode === 200) {
            try {
              const response = JSON.parse(data);
              const uploadUrls = response.upload_urls || {};
              const urlMap = new Map<number, string>();

              // Gateway returns sequences as strings in JSON, convert to int keys
              for (const [seqStr, url] of Object.entries(uploadUrls)) {
                urlMap.set(parseInt(seqStr, 10), url as string);
              }

              resolve(urlMap);
            } catch {
              this.logger?.warn('Failed to parse screenshot upload URLs response');
              resolve(new Map());
            }
          } else {
            this.logger?.warn(`Failed to get screenshot URLs: HTTP ${res.statusCode}`);
            resolve(new Map());
          }
        });
      });

      req.on('error', error => {
        this.logger?.warn(`Error requesting screenshot URLs: ${error.message}`);
        resolve(new Map());
      });

      req.on('timeout', () => {
        req.destroy();
        this.logger?.warn('Screenshot URLs request timeout');
        resolve(new Map());
      });

      req.write(body);
      req.end();
    });
  }

  /**
   * Upload screenshots extracted from trace events.
   *
   * Steps:
   * 1. Request pre-signed URLs from gateway (/v1/screenshots/init)
   * 2. Decode base64 to image bytes
   * 3. Upload screenshots in parallel (10 concurrent workers)
   * 4. Track upload progress
   *
   * @param screenshots - Map of sequence to screenshot data
   */
  private async _uploadScreenshots(
    screenshots: Map<number, { base64: string; format: string; stepId?: string }>
  ): Promise<void> {
    if (screenshots.size === 0) {
      return;
    }

    // 1. Request pre-signed URLs from gateway
    const sequences = Array.from(screenshots.keys()).sort((a, b) => a - b);
    const uploadUrls = await this._requestScreenshotUrls(sequences);

    if (uploadUrls.size === 0) {
      this.logger?.warn(
        'No screenshot upload URLs received, skipping upload. This may indicate API key permission issue, gateway error, or network problem.'
      );
      return;
    }

    // 2. Upload screenshots in parallel
    const uploadPromises: Promise<boolean>[] = [];
    const uploadSequences: number[] = [];

    uploadUrls.forEach((url, seq) => {
      const screenshotData = screenshots.get(seq);
      if (!screenshotData) {
        return;
      }

      uploadSequences.push(seq);
      const uploadPromise = this._uploadSingleScreenshot(seq, url, screenshotData);
      uploadPromises.push(uploadPromise);
    });

    // Wait for all uploads (max 10 concurrent)
    const results = await Promise.allSettled(uploadPromises.slice(0, 10));

    // Process remaining uploads in batches of 10
    for (let i = 10; i < uploadPromises.length; i += 10) {
      const batch = uploadPromises.slice(i, i + 10);
      const batchResults = await Promise.allSettled(batch);
      results.push(...batchResults);
    }

    // Count successes and failures
    let uploadedCount = 0;
    const failedSequences: number[] = [];

    for (let i = 0; i < results.length; i++) {
      const result = results[i];
      if (result.status === 'fulfilled' && result.value) {
        uploadedCount++;
      } else {
        failedSequences.push(uploadSequences[i]);
      }
    }

    // 3. Report results
    const totalCount = uploadUrls.size;
    if (uploadedCount === totalCount) {
      const totalSizeMB = this.screenshotTotalSizeBytes / 1024 / 1024;
      if (this.logger) {
        this.logger.info(
          `All ${totalCount} screenshots uploaded successfully (total size: ${totalSizeMB.toFixed(2)} MB)`
        );
      }
    } else {
      if (this.logger) {
        this.logger.warn(
          `Uploaded ${uploadedCount}/${totalCount} screenshots. Failed sequences: ${failedSequences.length > 0 ? failedSequences.join(', ') : 'none'}`
        );
      }
    }
  }

  /**
   * Upload a single screenshot to pre-signed URL.
   *
   * @param sequence - Screenshot sequence number
   * @param uploadUrl - Pre-signed upload URL
   * @param screenshotData - Screenshot data with base64 and format
   * @returns True if upload successful, false otherwise
   */
  private async _uploadSingleScreenshot(
    sequence: number,
    uploadUrl: string,
    screenshotData: { base64: string; format: string; stepId?: string }
  ): Promise<boolean> {
    try {
      // Decode base64 to image bytes
      const imageBytes = Buffer.from(screenshotData.base64, 'base64');
      const imageSize = imageBytes.length;

      // Update total size
      this.screenshotTotalSizeBytes += imageSize;

      // Upload to pre-signed URL
      const statusCode = await this._uploadScreenshotToCloud(
        uploadUrl,
        imageBytes,
        screenshotData.format as 'png' | 'jpeg'
      );

      if (statusCode === 200) {
        return true;
      } else {
        this.logger?.warn(`Screenshot ${sequence} upload failed: HTTP ${statusCode}`);
        return false;
      }
    } catch (error: any) {
      this.logger?.warn(`Screenshot ${sequence} upload error: ${error.message}`);
      return false;
    }
  }

  /**
   * Upload screenshot data to cloud using pre-signed URL
   */
  private async _uploadScreenshotToCloud(
    uploadUrl: string,
    data: Buffer,
    format: 'png' | 'jpeg'
  ): Promise<number> {
    return new Promise((resolve, reject) => {
      const url = new URL(uploadUrl);
      const protocol = url.protocol === 'https:' ? https : http;

      const options = {
        hostname: url.hostname,
        port: url.port || (url.protocol === 'https:' ? 443 : 80),
        path: url.pathname + url.search,
        method: 'PUT',
        headers: {
          'Content-Type': `image/${format}`,
          'Content-Length': data.length,
        },
        timeout: 30000, // 30 second timeout per screenshot
      };

      const req = protocol.request(options, res => {
        res.on('data', () => {});
        res.on('end', () => {
          resolve(res.statusCode || 500);
        });
      });

      req.on('error', error => {
        reject(error);
      });

      req.on('timeout', () => {
        req.destroy();
        reject(new Error('Screenshot upload timeout'));
      });

      req.write(data);
      req.end();
    });
  }

  /**
   * Delete local files after successful upload.
   */
  private async _cleanupFiles(): Promise<void> {
    // Delete trace file
    try {
      if (fs.existsSync(this.tempFilePath)) {
        await fsPromises.unlink(this.tempFilePath);
      }
    } catch {
      // Ignore cleanup errors
    }
  }

  /**
   * Get unique identifier for this sink
   */
  getSinkType(): string {
    return `CloudTraceSink(${this.uploadUrl.substring(0, 50)}...)`;
  }
}
