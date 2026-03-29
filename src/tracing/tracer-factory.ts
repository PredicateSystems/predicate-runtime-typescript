/**
 * Tracer Factory with Automatic Tier Detection
 *
 * Provides convenient factory function for creating tracers with cloud upload support
 *
 * PRODUCTION HARDENING:
 * - Recovers orphaned traces from previous crashes on SDK init (Risk #3)
 * - Passes runId to CloudTraceSink for persistent cache naming (Risk #1)
 * - Auto-closes tracers on process exit to prevent data loss (atexit safety net)
 */

import * as path from 'path';
import * as fs from 'fs';
import * as os from 'os';
import * as https from 'https';
import * as http from 'http';
import { URL } from 'url';
import { randomUUID } from 'crypto';
import { Tracer } from './tracer';
import { CloudTraceSink, SentienceLogger } from './cloud-sink';
import { JsonlTraceSink } from './jsonl-sink';

// ============================================================================
// Process Exit Cleanup (atexit safety net)
// ============================================================================

/**
 * Global registry of active tracers for process exit cleanup.
 * Uses Map with tracer ID as key to allow proper cleanup on close.
 *
 * Note: We store the tracer directly (not WeakRef) because:
 * 1. Tracers are explicitly unregistered when close() is called
 * 2. We need the tracer reference available during process exit
 * 3. If a tracer is garbage collected without close(), we want to close it on exit
 */
const activeTracers: Map<number, Tracer> = new Map();
let nextTracerId = 0;
let processExitHandlerRegistered = false;

/**
 * Cleanup handler called on process exit.
 * Closes all active tracers to ensure trace data is uploaded to cloud.
 * This prevents data loss when users forget to call tracer.close().
 */
function cleanupTracersOnExit(): void {
  for (const [tracerId, tracer] of activeTracers.entries()) {
    if (tracer && !tracer.isClosed()) {
      try {
        // Use non-blocking close for exit handler
        // Note: We can't await in exit handlers, so we do best-effort
        tracer.close(false).catch(() => {
          // Best effort - don't throw during exit
        });
      } catch {
        // Best effort - don't throw during exit
      }
    }
    activeTracers.delete(tracerId);
  }
}

/**
 * Register a tracer for automatic cleanup on process exit.
 * @param tracer - Tracer instance to register
 */
function registerTracerForCleanup(tracer: Tracer): void {
  const tracerId = nextTracerId++;
  activeTracers.set(tracerId, tracer);

  // Set callback on tracer so it unregisters itself when closed
  tracer._onCloseCallback = () => {
    activeTracers.delete(tracerId);
  };

  // Register process exit handlers on first tracer creation
  if (!processExitHandlerRegistered) {
    // Handle normal exit
    process.on('exit', cleanupTracersOnExit);
    // Handle Ctrl+C
    process.on('SIGINT', () => {
      cleanupTracersOnExit();
      process.exit(130);
    });
    // Handle kill
    process.on('SIGTERM', () => {
      cleanupTracersOnExit();
      process.exit(143);
    });
    // Handle uncaught exceptions (best effort)
    process.on('uncaughtException', error => {
      console.error('Uncaught exception:', error);
      cleanupTracersOnExit();
      process.exit(1);
    });

    processExitHandlerRegistered = true;
  }
}

/**
 * Helper to emit run_start event with available metadata
 */
function emitRunStart(
  tracer: Tracer,
  agentType?: string,
  llmModel?: string,
  goal?: string,
  startUrl?: string
): void {
  try {
    const config: Record<string, string> = {};
    if (goal) {
      config.goal = goal;
    }
    if (startUrl) {
      config.start_url = startUrl;
    }

    tracer.emitRunStart(
      agentType || 'SentienceAgent',
      llmModel,
      Object.keys(config).length > 0 ? config : undefined
    );
  } catch {
    // Tracing must be non-fatal
  }
}

/**
 * Sentience API base URL (constant)
 */
export const SENTIENCE_API_URL = 'https://api.sentienceapi.com';

/**
 * Get persistent cache directory for traces
 */
function getPersistentCacheDir(): string {
  const homeDir = os.homedir();
  return path.join(homeDir, '.sentience', 'traces', 'pending');
}

/**
 * Recover orphaned traces from previous crashes
 * PRODUCTION FIX: Risk #3 - Upload traces from crashed sessions
 *
 * Note: Silently skips in test environments to avoid test noise
 */
async function recoverOrphanedTraces(
  apiKey: string,
  apiUrl: string = SENTIENCE_API_URL
): Promise<void> {
  // Skip orphan recovery in test environments (CI, Jest, etc.)
  // This prevents test failures from orphan recovery attempts
  const isTestEnv =
    process.env.CI === 'true' ||
    process.env.NODE_ENV === 'test' ||
    process.env.JEST_WORKER_ID !== undefined ||
    (typeof global !== 'undefined' && (global as any).__JEST__);

  if (isTestEnv) {
    return;
  }

  const cacheDir = getPersistentCacheDir();

  if (!fs.existsSync(cacheDir)) {
    return;
  }

  let orphanedFiles: string[];
  try {
    orphanedFiles = fs.readdirSync(cacheDir).filter(f => f.endsWith('.jsonl'));
  } catch {
    // Silently fail if directory read fails (permissions, etc.)
    return;
  }

  if (orphanedFiles.length === 0) {
    return;
  }

  console.log(
    `⚠️  [Sentience] Found ${orphanedFiles.length} un-uploaded trace(s) from previous run(s)`
  );
  console.log('   Attempting to upload now...');

  for (const file of orphanedFiles) {
    const filePath = path.join(cacheDir, file);
    const runId = path.basename(file, '.jsonl');

    try {
      // Request upload URL for this orphaned trace
      // Use a shorter timeout for orphan recovery to avoid blocking
      const response = await Promise.race([
        httpPost(
          `${apiUrl}/v1/traces/init`,
          { run_id: runId },
          { Authorization: `Bearer ${apiKey}` }
        ),
        new Promise<{ status: number; data: any }>(resolve =>
          setTimeout(() => resolve({ status: 500, data: {} }), 5000)
        ),
      ]);

      if (response.status === 200 && response.data.upload_url) {
        // Create a temporary CloudTraceSink to upload this orphaned trace
        const sink = new CloudTraceSink(response.data.upload_url, runId);
        await sink.close(); // This will upload the existing file
        console.log(`✅ [Sentience] Uploaded orphaned trace: ${runId}`);
      } else if (response.status === 409) {
        // HTTP 409 means trace already exists (already uploaded)
        // Treat as success and delete local file
        console.log(`✅ [Sentience] Trace ${runId} already exists in cloud (skipping re-upload)`);
        try {
          fs.unlinkSync(filePath);
        } catch {
          // Ignore cleanup errors
        }
      }
      // Silently skip other failures - don't log errors for orphan recovery
      // These are expected in many scenarios (network issues, invalid API keys, etc.)
    } catch {
      // Silently skip failures - don't log errors for orphan recovery
      // These are expected in many scenarios (network issues, invalid API keys, etc.)
    }
  }
}

/**
 * Make HTTP/HTTPS POST request using built-in Node modules
 */
function httpPost(
  url: string,
  data: any,
  headers: Record<string, string>
): Promise<{
  status: number;
  data: any;
}> {
  return new Promise((resolve, reject) => {
    const urlObj = new URL(url);
    const protocol = urlObj.protocol === 'https:' ? https : http;

    const body = JSON.stringify(data);

    const options = {
      hostname: urlObj.hostname,
      port: urlObj.port || (urlObj.protocol === 'https:' ? 443 : 80),
      path: urlObj.pathname + urlObj.search,
      method: 'POST',
      headers: {
        'Content-Type': 'application/json',
        'Content-Length': Buffer.byteLength(body),
        ...headers,
      },
      timeout: 10000, // 10 second timeout
    };

    const req = protocol.request(options, res => {
      let responseBody = '';

      res.on('data', chunk => {
        responseBody += chunk;
      });

      res.on('end', () => {
        try {
          const parsed = responseBody ? JSON.parse(responseBody) : {};
          resolve({ status: res.statusCode || 500, data: parsed });
        } catch {
          resolve({ status: res.statusCode || 500, data: {} });
        }
      });
    });

    req.on('error', error => {
      reject(error);
    });

    req.on('timeout', () => {
      req.destroy();
      reject(new Error('Request timeout'));
    });

    req.write(body);
    req.end();
  });
}

/**
 * Create tracer with automatic tier detection
 *
 * Tier Detection:
 * - If apiKey is provided AND uploadTrace is true: Try to initialize CloudTraceSink (Pro/Enterprise)
 * - If cloud init fails, no apiKey, or uploadTrace is false: Fall back to JsonlTraceSink (Free tier)
 *
 * @param options - Configuration options
 * @param options.apiKey - Sentience API key (e.g., "sk_pro_xxxxx")
 * @param options.runId - Unique identifier for this agent run (generates UUID if not provided)
 * @param options.apiUrl - Sentience API base URL (default: https://api.sentienceapi.com)
 * @param options.logger - Optional logger instance for logging file sizes and errors
 * @param options.uploadTrace - Enable cloud trace upload (default: true for backward compatibility)
 * @param options.goal - User's goal/objective for this trace run. This will be displayed as the trace name in the frontend. Should be descriptive and action-oriented. Example: "Add wireless headphones to cart on Amazon"
 * @param options.agentType - Type of agent running (e.g., "SentienceAgent", "CustomAgent")
 * @param options.llmModel - LLM model used (e.g., "gpt-4-turbo", "claude-3-5-sonnet")
 * @param options.startUrl - Starting URL of the agent run (e.g., "https://amazon.com")
 * @param options.screenshotProcessor - Optional function to process screenshots before upload. Takes base64 string, returns processed base64 string. Useful for PII redaction or custom image processing.
 * @param options.autoEmitRunStart - If true (default), automatically emit run_start event with provided metadata. This ensures traces have complete structure for Studio visualization.
 * @returns Tracer configured with appropriate sink
 *
 * @example
 * ```typescript
 * // Pro tier user with goal and metadata
 * const tracer = await createTracer({
 *   apiKey: "sk_pro_xyz",
 *   runId: "demo",
 *   goal: "Add headphones to cart",
 *   agentType: "SentienceAgent",
 *   llmModel: "gpt-4-turbo",
 *   startUrl: "https://amazon.com",
 *   uploadTrace: true
 * });
 * // Returns: Tracer with CloudTraceSink
 * // run_start event is automatically emitted
 *
 * // With screenshot processor for PII redaction
 * const redactPII = (screenshot: string): string => {
 *   // Your custom redaction logic
 *   return redactedScreenshot;
 * };
 * const tracer = await createTracer({
 *   apiKey: "sk_pro_xyz",
 *   screenshotProcessor: redactPII
 * });
 * // Screenshots will be processed before upload
 *
 * // Pro tier user with local-only tracing
 * const tracer = await createTracer({ apiKey: "sk_pro_xyz", runId: "demo", uploadTrace: false });
 * // Returns: Tracer with JsonlTraceSink (local-only)
 *
 * // Disable auto-emit for manual control
 * const tracer = await createTracer({ runId: "demo", autoEmitRunStart: false });
 * tracer.emitRunStart("MyAgent", "gpt-4o"); // Manual emit
 *
 * // Free tier user
 * const tracer = await createTracer({ runId: "demo" });
 * // Returns: Tracer with JsonlTraceSink (local-only)
 *
 * // Use with agent
 * const agent = new SentienceAgent(browser, llm, 50, true, tracer);
 * await agent.act("Click search");
 * await tracer.close(); // Uploads to cloud if uploadTrace: true and Pro tier
 * ```
 */
export async function createTracer(options: {
  apiKey?: string;
  runId?: string;
  apiUrl?: string;
  logger?: SentienceLogger;
  uploadTrace?: boolean;
  goal?: string;
  agentType?: string;
  llmModel?: string;
  startUrl?: string;
  screenshotProcessor?: (screenshot: string) => string;
  autoEmitRunStart?: boolean;
}): Promise<Tracer> {
  const runId = options.runId || randomUUID();
  const apiUrl = options.apiUrl || SENTIENCE_API_URL;
  // Default uploadTrace to true for backward compatibility
  const uploadTrace = options.uploadTrace !== false;

  // PRODUCTION FIX: Recover orphaned traces from previous crashes
  // Note: This is skipped in test environments (see recoverOrphanedTraces function)
  // Run in background to avoid blocking tracer creation
  // Only recover if uploadTrace is enabled
  if (options.apiKey && uploadTrace) {
    // Don't await - run in background to avoid blocking
    recoverOrphanedTraces(options.apiKey, apiUrl).catch(() => {
      // Silently fail - orphan recovery should not block tracer creation
    });
  }

  // 1. Try to initialize Cloud Sink (Pro/Enterprise tier)
  // Only attempt cloud init if uploadTrace is enabled
  if (options.apiKey && uploadTrace) {
    try {
      // Build metadata object for trace initialization
      // Only include non-empty fields to avoid sending empty strings
      const metadata: Record<string, string> = {};
      if (options.goal && options.goal.trim()) {
        metadata.goal = options.goal.trim();
      }
      if (options.agentType && options.agentType.trim()) {
        metadata.agent_type = options.agentType.trim();
      }
      if (options.llmModel && options.llmModel.trim()) {
        metadata.llm_model = options.llmModel.trim();
      }
      if (options.startUrl && options.startUrl.trim()) {
        metadata.start_url = options.startUrl.trim();
      }

      // Build request payload
      const payload: Record<string, any> = { run_id: runId };
      if (Object.keys(metadata).length > 0) {
        payload.metadata = metadata;
      }

      // Request pre-signed upload URL from backend
      const response = await httpPost(`${apiUrl}/v1/traces/init`, payload, {
        Authorization: `Bearer ${options.apiKey}`,
      });

      if (response.status === 200 && response.data.upload_url) {
        const uploadUrl = response.data.upload_url;

        console.log('☁️  [Sentience] Cloud tracing enabled (Pro tier)');
        // PRODUCTION FIX: Pass runId for persistent cache naming
        const tracer = new Tracer(
          runId,
          new CloudTraceSink(uploadUrl, runId, options.apiKey, apiUrl, options.logger),
          options.screenshotProcessor
        );

        // Register for process exit cleanup (safety net for forgotten close())
        registerTracerForCleanup(tracer);

        // Auto-emit run_start for complete trace structure
        if (options.autoEmitRunStart !== false) {
          emitRunStart(tracer, options.agentType, options.llmModel, options.goal, options.startUrl);
        }

        return tracer;
      } else if (response.status === 403) {
        console.log('⚠️  [Sentience] Cloud tracing requires Pro tier');
        console.log('   Falling back to local-only tracing');
      } else {
        console.log(`⚠️  [Sentience] Cloud init failed: HTTP ${response.status}`);
        console.log('   Falling back to local-only tracing');
      }
    } catch (error: any) {
      if (error.message?.includes('timeout')) {
        console.log('⚠️  [Sentience] Cloud init timeout');
      } else if (error.code === 'ECONNREFUSED' || error.message?.includes('connect')) {
        console.log('⚠️  [Sentience] Cloud init connection error');
      } else {
        console.log(`⚠️  [Sentience] Cloud init error: ${error.message}`);
      }
      console.log('   Falling back to local-only tracing');
    }
  }

  // 2. Fallback to Local Sink (Free tier / Offline mode)
  const tracesDir = path.join(process.cwd(), 'traces');

  // Create traces directory if it doesn't exist
  if (!fs.existsSync(tracesDir)) {
    fs.mkdirSync(tracesDir, { recursive: true });
  }

  const localPath = path.join(tracesDir, `${runId}.jsonl`);
  console.log(`💾 [Sentience] Local tracing: ${localPath}`);

  const tracer = new Tracer(runId, new JsonlTraceSink(localPath), options.screenshotProcessor);

  // Register for process exit cleanup (ensures file is properly closed)
  registerTracerForCleanup(tracer);

  // Auto-emit run_start for complete trace structure
  if (options.autoEmitRunStart !== false) {
    emitRunStart(tracer, options.agentType, options.llmModel, options.goal, options.startUrl);
  }

  return tracer;
}

/**
 * Synchronous version of createTracer for non-async contexts
 * Always returns local JsonlTraceSink (no cloud upload)
 *
 * @param runId - Unique identifier for this agent run (generates UUID if not provided)
 * @returns Tracer with JsonlTraceSink
 */
export function createLocalTracer(runId?: string): Tracer {
  const traceRunId = runId || randomUUID();
  const tracesDir = path.join(process.cwd(), 'traces');

  if (!fs.existsSync(tracesDir)) {
    fs.mkdirSync(tracesDir, { recursive: true });
  }

  const localPath = path.join(tracesDir, `${traceRunId}.jsonl`);
  console.log(`💾 [Sentience] Local tracing: ${localPath}`);

  const tracer = new Tracer(traceRunId, new JsonlTraceSink(localPath));

  // Register for process exit cleanup (ensures file is properly closed)
  registerTracerForCleanup(tracer);

  return tracer;
}
