/**
 * Tests for screenshot extraction and upload in CloudTraceSink
 */

import * as fs from 'fs';
import * as path from 'path';
import * as os from 'os';
import { CloudTraceSink, SentienceLogger } from '../../src/tracing/cloud-sink';

// Mock logger for testing
class MockLogger implements SentienceLogger {
  public logs: string[] = [];

  info(message: string): void {
    this.logs.push(`[INFO] ${message}`);
  }

  warn(message: string): void {
    this.logs.push(`[WARN] ${message}`);
  }

  error(message: string): void {
    this.logs.push(`[ERROR] ${message}`);
  }
}

describe('Screenshot Extraction and Upload', () => {
  let uploadUrl: string;
  let runId: string;
  let cacheDir: string;

  beforeEach(() => {
    uploadUrl = 'https://sentience.nyc3.digitaloceanspaces.com/user123/run456/trace.jsonl.gz';
    runId = `test-screenshot-${Date.now()}`;
    cacheDir = path.join(os.homedir(), '.sentience', 'traces', 'pending');
  });

  afterEach(() => {
    // Cleanup test files
    const tracePath = path.join(cacheDir, `${runId}.jsonl`);
    const cleanedTracePath = path.join(cacheDir, `${runId}.cleaned.jsonl`);

    if (fs.existsSync(tracePath)) {
      fs.unlinkSync(tracePath);
    }

    if (fs.existsSync(cleanedTracePath)) {
      fs.unlinkSync(cleanedTracePath);
    }
  });

  describe('close(false)', () => {
    it('waits for the upload path when the returned promise is awaited', async () => {
      const sink = new CloudTraceSink(uploadUrl, runId);
      let uploadFinished = false;

      const uploadPromise = new Promise<void>(resolve => {
        setTimeout(() => {
          uploadFinished = true;
          resolve();
        }, 20);
      });

      const uploadSpy = jest
        .spyOn(sink as any, '_doUpload')
        .mockImplementation(() => uploadPromise);

      const closePromise = sink.close(false);

      expect(uploadFinished).toBe(false);

      await closePromise;

      expect(uploadFinished).toBe(true);
      expect(uploadSpy).toHaveBeenCalledTimes(1);

      uploadSpy.mockRestore();
    });

    it('still logs upload failures outside the Jest suppression path', async () => {
      const sink = new CloudTraceSink(uploadUrl, runId);
      const originalWorkerId = process.env.JEST_WORKER_ID;
      const consoleErrorSpy = jest.spyOn(console, 'error').mockImplementation(() => {});

      delete process.env.JEST_WORKER_ID;
      sink.emit({
        v: 1,
        type: 'action',
        ts: '2026-01-01T00:00:00.000Z',
        run_id: runId,
        seq: 1,
        data: {
          action: 'click',
          element_id: 123,
        },
      });

      const uploadSpy = jest.spyOn(sink as any, '_uploadToCloud').mockResolvedValue(500);

      try {
        await sink.close(false);
        expect(consoleErrorSpy).toHaveBeenCalledWith('❌ [Sentience] Upload failed: HTTP 500');
        expect(consoleErrorSpy).toHaveBeenCalledWith(
          expect.stringContaining('Local trace preserved at:')
        );
      } finally {
        if (originalWorkerId === undefined) {
          delete process.env.JEST_WORKER_ID;
        } else {
          process.env.JEST_WORKER_ID = originalWorkerId;
        }
        consoleErrorSpy.mockRestore();
        uploadSpy.mockRestore();
      }
    });
  });

  describe('_extractScreenshotsFromTrace', () => {
    it('should extract screenshots from trace events', async () => {
      const sink = new CloudTraceSink(uploadUrl, runId);

      // Create a trace file with screenshot events
      const testImageBase64 =
        'iVBORw0KGgoAAAANSUhEUgAAAAEAAAABCAYAAAAfFcSJAAAADUlEQVR42mNk+M9QDwADhgGAWjR9awAAAABJRU5ErkJggg==';

      // Emit a snapshot event with screenshot
      sink.emit({
        v: 1,
        type: 'snapshot',
        ts: '2026-01-01T00:00:00.000Z',
        run_id: runId,
        seq: 1,
        step_id: 'step-1',
        data: {
          url: 'https://example.com',
          element_count: 10,
          screenshot_base64: testImageBase64,
          screenshot_format: 'png',
        },
      });

      // Close to write file
      await sink.close(false);

      // Wait a bit for file to be written
      await new Promise(resolve => setTimeout(resolve, 100));

      // Extract screenshots
      const screenshots = await (sink as any)._extractScreenshotsFromTrace();

      expect(screenshots.size).toBe(1);
      expect(screenshots.get(1)).toBeDefined();
      expect(screenshots.get(1)?.base64).toBe(testImageBase64);
      expect(screenshots.get(1)?.format).toBe('png');
      expect(screenshots.get(1)?.stepId).toBe('step-1');
    });

    it('should handle multiple screenshots', async () => {
      const sink = new CloudTraceSink(uploadUrl, runId);
      const testImageBase64 =
        'iVBORw0KGgoAAAANSUhEUgAAAAEAAAABCAYAAAAfFcSJAAAADUlEQVR42mNk+M9QDwADhgGAWjR9awAAAABJRU5ErkJggg==';

      // Emit multiple snapshot events with screenshots
      for (let i = 1; i <= 3; i++) {
        sink.emit({
          v: 1,
          type: 'snapshot',
          ts: '2026-01-01T00:00:00.000Z',
          run_id: runId,
          seq: i,
          step_id: `step-${i}`,
          data: {
            url: 'https://example.com',
            element_count: 10,
            screenshot_base64: testImageBase64,
            screenshot_format: 'png',
          },
        });
      }

      await sink.close(false);
      await new Promise(resolve => setTimeout(resolve, 100));

      const screenshots = await (sink as any)._extractScreenshotsFromTrace();
      expect(screenshots.size).toBe(3);
    });

    it('should skip events without screenshots', async () => {
      const sink = new CloudTraceSink(uploadUrl, runId);

      // Emit snapshot without screenshot
      sink.emit({
        v: 1,
        type: 'snapshot',
        ts: '2026-01-01T00:00:00.000Z',
        run_id: runId,
        seq: 1,
        data: {
          url: 'https://example.com',
          element_count: 10,
          // No screenshot_base64
        },
      });

      await sink.close(false);
      await new Promise(resolve => setTimeout(resolve, 100));

      const screenshots = await (sink as any)._extractScreenshotsFromTrace();
      expect(screenshots.size).toBe(0);
    });
  });

  describe('_createCleanedTrace', () => {
    it('should remove screenshot_base64 from events', async () => {
      const sink = new CloudTraceSink(uploadUrl, runId);
      const testImageBase64 =
        'iVBORw0KGgoAAAANSUhEUgAAAAEAAAABCAYAAAAfFcSJAAAADUlEQVR42mNk+M9QDwADhgGAWjR9awAAAABJRU5ErkJggg==';

      // Emit snapshot event with screenshot
      sink.emit({
        v: 1,
        type: 'snapshot',
        ts: '2026-01-01T00:00:00.000Z',
        run_id: runId,
        seq: 1,
        data: {
          url: 'https://example.com',
          element_count: 10,
          screenshot_base64: testImageBase64,
          screenshot_format: 'png',
        },
      });

      await sink.close(false);
      await new Promise(resolve => setTimeout(resolve, 100));

      // Create cleaned trace
      const cleanedTracePath = path.join(cacheDir, `${runId}.cleaned.jsonl`);
      await (sink as any)._createCleanedTrace(cleanedTracePath);

      // Read cleaned trace
      const cleanedContent = fs.readFileSync(cleanedTracePath, 'utf-8');
      const cleanedEvent = JSON.parse(cleanedContent.trim());

      // Verify screenshot fields are removed
      expect(cleanedEvent.data.screenshot_base64).toBeUndefined();
      expect(cleanedEvent.data.screenshot_format).toBeUndefined();
      expect(cleanedEvent.data.url).toBe('https://example.com');
      expect(cleanedEvent.data.element_count).toBe(10);
    });

    it('should preserve other event types unchanged', async () => {
      const sink = new CloudTraceSink(uploadUrl, runId);

      // Emit non-snapshot event
      sink.emit({
        v: 1,
        type: 'action',
        ts: '2026-01-01T00:00:00.000Z',
        run_id: runId,
        seq: 1,
        data: {
          action: 'click',
          element_id: 123,
        },
      });

      await sink.close(false);
      await new Promise(resolve => setTimeout(resolve, 100));

      const cleanedTracePath = path.join(cacheDir, `${runId}.cleaned.jsonl`);
      await (sink as any)._createCleanedTrace(cleanedTracePath);

      const cleanedContent = fs.readFileSync(cleanedTracePath, 'utf-8');
      const cleanedEvent = JSON.parse(cleanedContent.trim());

      // Verify action event is unchanged
      expect(cleanedEvent.type).toBe('action');
      expect(cleanedEvent.data.action).toBe('click');
      expect(cleanedEvent.data.element_id).toBe(123);
    });
  });

  describe('_requestScreenshotUrls', () => {
    it('should request URLs from gateway', async () => {
      const apiKey = 'sk_test_123';
      const sink = new CloudTraceSink(uploadUrl, runId, apiKey);

      // Mock HTTP request
      const originalRequest = require('https').request;
      const mockUrls = {
        '1': 'https://sentience.nyc3.digitaloceanspaces.com/user123/run456/screenshots/step_0001.png?signature=...',
        '2': 'https://sentience.nyc3.digitaloceanspaces.com/user123/run456/screenshots/step_0002.png?signature=...',
      };

      let requestCalled = false;
      require('https').request = jest.fn((options: any, callback: any) => {
        requestCalled = true;
        const mockRes = {
          statusCode: 200,
          on: jest.fn((event: string, handler: any) => {
            if (event === 'data') {
              handler(JSON.stringify({ upload_urls: mockUrls }));
            } else if (event === 'end') {
              handler();
            }
          }),
        };
        setTimeout(() => callback(mockRes), 0);
        return {
          write: jest.fn(),
          end: jest.fn(),
          on: jest.fn(),
        };
      });

      const result = await (sink as any)._requestScreenshotUrls([1, 2]);

      expect(requestCalled).toBe(true);
      expect(result.size).toBe(2);
      expect(result.get(1)).toBe(mockUrls['1']);
      expect(result.get(2)).toBe(mockUrls['2']);

      require('https').request = originalRequest;
      sink.close(false);
    });

    it('should return empty map on failure', async () => {
      const apiKey = 'sk_test_123';
      const sink = new CloudTraceSink(uploadUrl, runId, apiKey);

      // Mock HTTP request with failure
      const originalRequest = require('https').request;
      require('https').request = jest.fn((options: any, callback: any) => {
        const mockRes = {
          statusCode: 500,
          on: jest.fn((event: string, handler: any) => {
            if (event === 'end') {
              handler();
            }
          }),
        };
        setTimeout(() => callback(mockRes), 0);
        return {
          write: jest.fn(),
          end: jest.fn(),
          on: jest.fn(),
        };
      });

      const result = await (sink as any)._requestScreenshotUrls([1, 2]);

      expect(result.size).toBe(0);

      require('https').request = originalRequest;
      sink.close(false);
    });
  });
});
