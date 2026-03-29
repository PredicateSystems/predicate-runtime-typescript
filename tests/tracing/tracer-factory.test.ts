/**
 * Tests for Tracer Factory Functions
 */

import * as fs from 'fs';
import * as path from 'path';
import * as http from 'http';
import { createTracer, createLocalTracer } from '../../src/tracing/tracer-factory';
import { CloudTraceSink } from '../../src/tracing/cloud-sink';
import { JsonlTraceSink } from '../../src/tracing/jsonl-sink';

describe('createTracer', () => {
  let mockServer: http.Server;
  let serverPort: number;
  let apiUrl: string;
  const testTracesDir = path.join(process.cwd(), 'traces');

  // Start a mock HTTP server before tests
  beforeAll(done => {
    mockServer = http.createServer((req, res) => {
      // Store request info for verification
      (mockServer as any).lastRequest = {
        method: req.method,
        url: req.url,
        headers: req.headers,
      };

      // Read request body
      const chunks: Buffer[] = [];
      req.on('data', chunk => chunks.push(chunk));
      req.on('end', () => {
        (mockServer as any).lastRequestBody = Buffer.concat(chunks);

        // Parse and respond based on endpoint
        if (req.url === '/v1/traces/init' && req.method === 'POST') {
          const authorization = req.headers['authorization'];

          // Check for special test conditions (only for valid auth)
          if (authorization && (mockServer as any).shouldTimeout) {
            // Simulate timeout - don't respond (connection will timeout)
            return; // Don't call res.end() - this will cause a timeout
          }

          if (authorization && (mockServer as any).shouldError) {
            // Simulate error - return 500 without upload_url
            res.writeHead(500);
            res.end(JSON.stringify({ error: 'Internal Server Error' }));
            return;
          }

          if (!authorization) {
            res.writeHead(401);
            res.end(JSON.stringify({ error: 'Unauthorized' }));
          } else if (authorization === 'Bearer sk_free_123') {
            // Free tier - return 403
            res.writeHead(403);
            res.end(JSON.stringify({ error: 'Pro tier required' }));
          } else if (authorization === 'Bearer sk_pro_123') {
            // Pro tier - return upload URL
            res.writeHead(200);
            res.end(JSON.stringify({ upload_url: `http://localhost:${serverPort}/upload` }));
          } else {
            res.writeHead(200);
            res.end(JSON.stringify({ upload_url: `http://localhost:${serverPort}/upload` }));
          }
        } else if (req.url === '/upload' && req.method === 'PUT') {
          // Mock upload endpoint
          res.writeHead(200);
          res.end('OK');
        } else {
          res.writeHead(404);
          res.end('Not Found');
        }
      });
    });

    mockServer.listen(0, () => {
      const address = mockServer.address();
      if (address && typeof address === 'object') {
        serverPort = address.port;
        apiUrl = `http://localhost:${serverPort}`;
        done();
      }
    });
  });

  afterAll(done => {
    mockServer.close(done);
  });

  beforeEach(() => {
    // Reset server state
    delete (mockServer as any).lastRequest;
    delete (mockServer as any).lastRequestBody;
    (mockServer as any).shouldTimeout = false;
    (mockServer as any).shouldError = false;

    // Create traces directory
    if (!fs.existsSync(testTracesDir)) {
      fs.mkdirSync(testTracesDir, { recursive: true });
    }
  });

  afterEach(() => {
    // Cleanup traces directory
    if (fs.existsSync(testTracesDir)) {
      const files = fs.readdirSync(testTracesDir);
      files.forEach(file => {
        const filePath = path.join(testTracesDir, file);
        if (fs.existsSync(filePath)) {
          fs.unlinkSync(filePath);
        }
      });
    }
  });

  describe('Pro tier cloud tracing', () => {
    it('should create CloudTraceSink for Pro tier with valid API key', async () => {
      const tracer = await createTracer({
        apiKey: 'sk_pro_123',
        runId: 'test-run',
        apiUrl: apiUrl,
      });

      expect(tracer).toBeDefined();
      expect(tracer.getRunId()).toBe('test-run');
      // Verify it's a CloudTraceSink by checking the sink type
      expect(tracer.getSinkType()).toContain('CloudTraceSink');

      // Verify API was called
      expect((mockServer as any).lastRequest).toBeDefined();
      expect((mockServer as any).lastRequest.method).toBe('POST');
      expect((mockServer as any).lastRequest.url).toBe('/v1/traces/init');
      expect((mockServer as any).lastRequest.headers['authorization']).toBe('Bearer sk_pro_123');

      await tracer.close();
    });

    it('should generate run_id if not provided', async () => {
      const tracer = await createTracer({
        apiKey: 'sk_pro_123',
        apiUrl: apiUrl,
      });

      expect(tracer).toBeDefined();
      const runId = tracer.getRunId();
      expect(runId).toBeDefined();
      expect(runId.length).toBe(36); // UUID format

      await tracer.close();
    });

    it('should use custom API URL', async () => {
      const tracer = await createTracer({
        apiKey: 'sk_pro_123',
        runId: 'test-run',
        apiUrl: apiUrl,
      });

      expect((mockServer as any).lastRequest.url).toBe('/v1/traces/init');

      await tracer.close();
    });
  });

  describe('Free tier fallback', () => {
    it('should fallback to JsonlTraceSink when no API key provided', async () => {
      const tracer = await createTracer({
        runId: 'test-run',
      });

      expect(tracer).toBeDefined();
      expect(tracer.getRunId()).toBe('test-run');
      // Verify it's a JsonlTraceSink by checking the sink type
      expect(tracer.getSinkType()).toContain('JsonlTraceSink');

      await tracer.close();
    });

    it('should fallback to JsonlTraceSink when API returns 403', async () => {
      const tracer = await createTracer({
        apiKey: 'sk_free_123',
        runId: 'test-run',
        apiUrl: apiUrl,
      });

      expect(tracer).toBeDefined();
      // Verify it's a JsonlTraceSink by checking the sink type
      expect(tracer.getSinkType()).toContain('JsonlTraceSink');

      await tracer.close();
    });

    it('should create local trace file in traces directory', async () => {
      const tracer = await createTracer({
        runId: 'test-run',
      });

      tracer.emitRunStart('TestAgent', 'gpt-4');
      await tracer.close();

      const traceFile = path.join(testTracesDir, 'test-run.jsonl');
      expect(fs.existsSync(traceFile)).toBe(true);
    });
  });

  describe('Error handling', () => {
    it('should fallback to local on API timeout', async () => {
      (mockServer as any).shouldTimeout = true;

      const tracer = await createTracer({
        apiKey: 'sk_pro_123',
        runId: 'test-run',
        apiUrl: apiUrl,
      });

      expect(tracer).toBeDefined();
      expect(tracer.getSinkType()).toContain('JsonlTraceSink');

      await tracer.close();
    });

    it('should fallback to local on API error', async () => {
      (mockServer as any).shouldError = true;

      const tracer = await createTracer({
        apiKey: 'sk_pro_123',
        runId: 'test-run',
        apiUrl: apiUrl,
      });

      expect(tracer).toBeDefined();
      expect(tracer.getSinkType()).toContain('JsonlTraceSink');

      await tracer.close();
    });

    it('should fallback to local on network error', async () => {
      const tracer = await createTracer({
        apiKey: 'sk_pro_123',
        runId: 'test-run',
        apiUrl: 'http://localhost:1/invalid', // Invalid port
      });

      expect(tracer).toBeDefined();
      expect(tracer.getSinkType()).toContain('JsonlTraceSink');

      expect(tracer).toBeDefined();
      expect(tracer.getSinkType()).toContain('JsonlTraceSink');

      await tracer.close();
    });

    it('should handle missing upload_url in API response', async () => {
      // Create a temporary server that returns 200 but no upload_url
      const tempServer = http.createServer((req, res) => {
        if (req.url === '/v1/traces/init') {
          res.writeHead(200);
          res.end(JSON.stringify({})); // No upload_url
        }
      });

      await new Promise<void>(resolve => {
        tempServer.listen(0, () => resolve());
      });

      const address = tempServer.address();
      if (address && typeof address === 'object') {
        const tempUrl = `http://localhost:${address.port}`;

        const tracer = await createTracer({
          apiKey: 'sk_pro_123',
          runId: 'test-run',
          apiUrl: tempUrl,
        });

        expect(tracer).toBeDefined();
        expect(tracer.getSinkType()).toContain('JsonlTraceSink');

        await tracer.close();
        tempServer.close();
      }
    });
  });

  describe('Integration', () => {
    it('should work with agent workflow (Pro tier)', async () => {
      const tracer = await createTracer({
        apiKey: 'sk_pro_123',
        runId: 'agent-test',
        apiUrl: apiUrl,
      });

      tracer.emitRunStart('SentienceAgent', 'gpt-4');
      tracer.emitStepStart('step-1', 1, 'Click button', 0, 'https://example.com');
      tracer.emit('custom_event', { data: 'test' } as any);
      tracer.emitRunEnd(1);

      await tracer.close();

      // Verify upload happened
      expect((mockServer as any).lastRequest).toBeDefined();
    });

    it('should work with agent workflow (Free tier)', async () => {
      const tracer = await createTracer({
        runId: 'agent-test',
        autoEmitRunStart: false, // Disable auto-emit since we're manually emitting
      });

      tracer.emitRunStart('SentienceAgent', 'gpt-4');
      tracer.emitStepStart('step-1', 1, 'Click button', 0, 'https://example.com');
      tracer.emitRunEnd(1);

      await tracer.close();

      // Verify local file was created
      const traceFile = path.join(testTracesDir, 'agent-test.jsonl');
      expect(fs.existsSync(traceFile)).toBe(true);

      // Verify content
      const content = fs.readFileSync(traceFile, 'utf-8');
      const lines = content.trim().split('\n');
      expect(lines.length).toBe(3); // run_start, step_start, run_end

      const event1 = JSON.parse(lines[0]);
      expect(event1.type).toBe('run_start');
    });
  });
});

describe('createLocalTracer', () => {
  const testTracesDir = path.join(process.cwd(), 'traces');

  beforeEach(() => {
    // Create traces directory
    if (!fs.existsSync(testTracesDir)) {
      fs.mkdirSync(testTracesDir, { recursive: true });
    }
  });

  afterEach(() => {
    // Cleanup traces directory
    if (fs.existsSync(testTracesDir)) {
      const files = fs.readdirSync(testTracesDir);
      files.forEach(file => {
        const filePath = path.join(testTracesDir, file);
        if (fs.existsSync(filePath)) {
          fs.unlinkSync(filePath);
        }
      });
    }
  });

  it('should always create JsonlTraceSink', () => {
    const tracer = createLocalTracer('test-run');

    expect(tracer).toBeDefined();
    expect(tracer.getRunId()).toBe('test-run');
    expect(tracer.getSinkType()).toContain('JsonlTraceSink');
  });

  it('should generate run_id if not provided', () => {
    const tracer = createLocalTracer();

    expect(tracer).toBeDefined();
    const runId = tracer.getRunId();
    expect(runId).toBeDefined();
    expect(runId.length).toBe(36); // UUID format
  });

  it('should create local trace file', async () => {
    const tracer = createLocalTracer('local-test');

    tracer.emitRunStart('TestAgent', 'gpt-4');
    tracer.emitRunEnd(1);

    await tracer.close();

    const traceFile = path.join(testTracesDir, 'local-test.jsonl');
    expect(fs.existsSync(traceFile)).toBe(true);
  });

  it('should work in synchronous contexts', () => {
    // This is the key use case - createLocalTracer is synchronous
    function syncFunction() {
      const tracer = createLocalTracer('sync-test');
      tracer.emitRunStart('SyncAgent', 'gpt-4');
      return tracer;
    }

    const tracer = syncFunction();
    expect(tracer).toBeDefined();
    expect(tracer.getSinkType()).toContain('JsonlTraceSink');
  });
});

describe('Tracer cleanup functionality', () => {
  const testTracesDir = path.join(process.cwd(), 'traces');

  beforeEach(() => {
    // Create traces directory
    if (!fs.existsSync(testTracesDir)) {
      fs.mkdirSync(testTracesDir, { recursive: true });
    }
  });

  afterEach(() => {
    // Cleanup traces directory
    if (fs.existsSync(testTracesDir)) {
      const files = fs.readdirSync(testTracesDir);
      files.forEach(file => {
        const filePath = path.join(testTracesDir, file);
        if (fs.existsSync(filePath)) {
          fs.unlinkSync(filePath);
        }
      });
    }
  });

  describe('isClosed method', () => {
    it('should return false for new tracer', () => {
      const tracer = createLocalTracer('test-closed-1');
      expect(tracer.isClosed()).toBe(false);
    });

    it('should return true after close is called', async () => {
      const tracer = createLocalTracer('test-closed-2');
      expect(tracer.isClosed()).toBe(false);

      await tracer.close();

      expect(tracer.isClosed()).toBe(true);
    });
  });

  describe('double-close prevention', () => {
    it('should prevent double close from causing issues', async () => {
      const tracer = createLocalTracer('test-double-close');
      tracer.emitRunStart('TestAgent', 'gpt-4');

      // First close should work
      await tracer.close();
      expect(tracer.isClosed()).toBe(true);

      // Second close should be a no-op (no error)
      await tracer.close();
      expect(tracer.isClosed()).toBe(true);
    });
  });

  describe('_onCloseCallback', () => {
    it('should call callback when close is invoked', async () => {
      const tracer = createLocalTracer('test-callback');
      let callbackInvoked = false;
      let callbackTracer: any = null;

      tracer._onCloseCallback = (t: any) => {
        callbackInvoked = true;
        callbackTracer = t;
      };

      expect(callbackInvoked).toBe(false);

      await tracer.close();

      expect(callbackInvoked).toBe(true);
      expect(callbackTracer).toBe(tracer);
    });

    it('should handle callback errors gracefully', async () => {
      const tracer = createLocalTracer('test-callback-error');

      tracer._onCloseCallback = () => {
        throw new Error('Callback error');
      };

      // Should not throw even when callback throws
      await expect(tracer.close()).resolves.not.toThrow();
      expect(tracer.isClosed()).toBe(true);
    });

    it('should not call callback on second close', async () => {
      const tracer = createLocalTracer('test-callback-once');
      let callCount = 0;

      tracer._onCloseCallback = () => {
        callCount++;
      };

      await tracer.close();
      expect(callCount).toBe(1);

      await tracer.close();
      expect(callCount).toBe(1); // Still 1, not called again
    });
  });

  describe('createTracer registers cleanup callback', () => {
    it('should set _onCloseCallback on created tracer', async () => {
      const tracer = await createTracer({
        runId: 'test-cleanup-registration',
      });

      // Factory should have set up the callback
      expect(tracer._onCloseCallback).toBeDefined();
      expect(typeof tracer._onCloseCallback).toBe('function');

      await tracer.close();
    });
  });

  describe('createLocalTracer registers cleanup callback', () => {
    it('should set _onCloseCallback on created tracer', () => {
      const tracer = createLocalTracer('test-local-cleanup');

      // Factory should have set up the callback
      expect(tracer._onCloseCallback).toBeDefined();
      expect(typeof tracer._onCloseCallback).toBe('function');
    });
  });
});
