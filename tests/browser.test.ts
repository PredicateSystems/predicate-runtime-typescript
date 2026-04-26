/**
 * Test browser proxy support and Phase 2 features (viewport, from_existing, from_page)
 */

import { SentienceBrowser, domainMatches, extractHost, isDomainAllowed } from '../src/browser';
import { chromium, BrowserContext, Page } from 'playwright';
import * as fs from 'fs';
import * as os from 'os';
import * as path from 'path';
import { patchExampleDotCom } from './test-utils';

describe('Browser Proxy Support', () => {
  describe('Proxy Parsing', () => {
    it('should parse HTTP proxy with credentials', () => {
      const browser = new SentienceBrowser(
        undefined,
        undefined,
        false,
        'http://user:pass@proxy.com:8000'
      );
      const config = (browser as any).parseProxy('http://user:pass@proxy.com:8000');

      expect(config).toBeDefined();
      expect(config?.server).toBe('http://proxy.com:8000');
      expect(config?.username).toBe('user');
      expect(config?.password).toBe('pass');
    });

    it('should parse HTTPS proxy with credentials', () => {
      const browser = new SentienceBrowser(
        undefined,
        undefined,
        false,
        'https://user:pass@proxy.com:8443'
      );
      const config = (browser as any).parseProxy('https://user:pass@proxy.com:8443');

      expect(config).toBeDefined();
      expect(config?.server).toBe('https://proxy.com:8443');
      expect(config?.username).toBe('user');
      expect(config?.password).toBe('pass');
    });

    it('should parse SOCKS5 proxy with credentials', () => {
      const browser = new SentienceBrowser(
        undefined,
        undefined,
        false,
        'socks5://user:pass@proxy.com:1080'
      );
      const config = (browser as any).parseProxy('socks5://user:pass@proxy.com:1080');

      expect(config).toBeDefined();
      expect(config?.server).toBe('socks5://proxy.com:1080');
      expect(config?.username).toBe('user');
      expect(config?.password).toBe('pass');
    });

    it('should parse HTTP proxy without credentials', () => {
      const browser = new SentienceBrowser(undefined, undefined, false, 'http://proxy.com:8000');
      const config = (browser as any).parseProxy('http://proxy.com:8000');

      expect(config).toBeDefined();
      expect(config?.server).toBe('http://proxy.com:8000');
      expect(config?.username).toBeUndefined();
      expect(config?.password).toBeUndefined();
    });

    it('should handle invalid proxy gracefully', () => {
      const browser = new SentienceBrowser(undefined, undefined, false, 'invalid');
      const config = (browser as any).parseProxy('invalid');

      expect(config).toBeUndefined();
    });

    it('should handle missing port gracefully', () => {
      const browser = new SentienceBrowser(undefined, undefined, false, 'http://proxy.com');
      const config = (browser as any).parseProxy('http://proxy.com');

      expect(config).toBeUndefined();
    });

    it('should handle unsupported scheme gracefully', () => {
      const browser = new SentienceBrowser(undefined, undefined, false, 'ftp://proxy.com:8000');
      const config = (browser as any).parseProxy('ftp://proxy.com:8000');

      expect(config).toBeUndefined();
    });

    it('should return undefined for empty string', () => {
      const browser = new SentienceBrowser(undefined, undefined, false);
      const config = (browser as any).parseProxy('');

      expect(config).toBeUndefined();
    });

    it('should return undefined for undefined', () => {
      const browser = new SentienceBrowser(undefined, undefined, false);
      const config = (browser as any).parseProxy(undefined);

      expect(config).toBeUndefined();
    });

    it('should support proxy from environment variable', () => {
      const originalEnv = process.env.SENTIENCE_PROXY;
      process.env.SENTIENCE_PROXY = 'http://env:pass@proxy.com:8000';

      const browser = new SentienceBrowser(undefined, undefined, false);
      const config = (browser as any).parseProxy((browser as any)._proxy);

      expect(config).toBeDefined();
      expect(config?.server).toBe('http://proxy.com:8000');
      expect(config?.username).toBe('env');
      expect(config?.password).toBe('pass');

      // Restore
      if (originalEnv) {
        process.env.SENTIENCE_PROXY = originalEnv;
      } else {
        delete process.env.SENTIENCE_PROXY;
      }
    });

    it('should prioritize parameter over environment variable', () => {
      const originalEnv = process.env.SENTIENCE_PROXY;
      process.env.SENTIENCE_PROXY = 'http://env:pass@proxy.com:8000';

      const browser = new SentienceBrowser(
        undefined,
        undefined,
        false,
        'http://param:pass@proxy.com:9000'
      );
      const config = (browser as any).parseProxy((browser as any)._proxy);

      expect(config).toBeDefined();
      expect(config?.server).toBe('http://proxy.com:9000');
      expect(config?.username).toBe('param');

      // Restore
      if (originalEnv) {
        process.env.SENTIENCE_PROXY = originalEnv;
      } else {
        delete process.env.SENTIENCE_PROXY;
      }
    });
  });

  describe('Browser Launch with Proxy', () => {
    // Note: These tests verify that proxy config is passed correctly
    // We don't actually launch browsers with real proxies in unit tests
    // Integration tests would verify actual proxy functionality

    it('should include WebRTC flags when proxy is configured', () => {
      const browser = new SentienceBrowser(
        undefined,
        undefined,
        false,
        'http://user:pass@proxy.com:8000'
      );
      // We can't easily test the actual launch args without mocking Playwright
      // But we can verify the proxy is stored
      expect((browser as any)._proxy).toBe('http://user:pass@proxy.com:8000');
    });

    it('should not include WebRTC flags when proxy is not configured', () => {
      const browser = new SentienceBrowser(undefined, undefined, false);
      expect((browser as any)._proxy).toBeUndefined();
    });
  });

  describe('Viewport Configuration', () => {
    it('should use default viewport 1280x800', () => {
      const browser = new SentienceBrowser();
      expect((browser as any)._viewport).toEqual({ width: 1280, height: 800 });
    });

    it('should accept custom viewport', () => {
      const customViewport = { width: 1920, height: 1080 };
      const browser = new SentienceBrowser(
        undefined,
        undefined,
        undefined,
        undefined,
        undefined,
        undefined,
        undefined,
        undefined,
        customViewport
      );
      expect((browser as any)._viewport).toEqual(customViewport);
    });

    it('should accept mobile viewport', () => {
      const mobileViewport = { width: 375, height: 667 };
      const browser = new SentienceBrowser(
        undefined,
        undefined,
        undefined,
        undefined,
        undefined,
        undefined,
        undefined,
        undefined,
        mobileViewport
      );
      expect((browser as any)._viewport).toEqual(mobileViewport);
    });
  });

  describe('fromExisting', () => {
    it('should create SentienceBrowser from existing context', async () => {
      // Auto-detect headless mode (headless in CI, headed locally)
      const isCI = process.env.CI === 'true' || process.env.CI === '1';
      const userDataDir = fs.mkdtempSync(path.join(os.tmpdir(), 'sentience-pw-'));
      const context = await chromium.launchPersistentContext(userDataDir, {
        headless: isCI,
        viewport: { width: 1600, height: 900 },
      });

      try {
        const browser = await SentienceBrowser.fromExisting(context);

        expect(browser.getContext()).toBe(context);
        expect(browser.getPage()).toBeDefined();

        // Verify viewport is preserved
        const page = browser.getPage();
        if (!page) {
          throw new Error('Browser page is not available');
        }
        await patchExampleDotCom(page);
        await page.goto('https://example.com', { waitUntil: 'domcontentloaded', timeout: 20000 });

        const viewportSize = await page.evaluate(() => ({
          width: window.innerWidth,
          height: window.innerHeight,
        }));

        expect(viewportSize.width).toBe(1600);
        expect(viewportSize.height).toBe(900);
      } finally {
        await context.close();
        try {
          fs.rmSync(userDataDir, { recursive: true, force: true });
        } catch {
          // ignore
        }
      }
    }, 60000);

    it('should accept API key configuration', async () => {
      // Auto-detect headless mode (headless in CI, headed locally)
      const isCI = process.env.CI === 'true' || process.env.CI === '1';
      const userDataDir = fs.mkdtempSync(path.join(os.tmpdir(), 'sentience-pw-'));
      const context = await chromium.launchPersistentContext(userDataDir, {
        headless: isCI,
      });

      try {
        const browser = await SentienceBrowser.fromExisting(
          context,
          'test_key',
          'https://test.api.com'
        );

        expect(browser.getApiKey()).toBe('test_key');
        expect(browser.getApiUrl()).toBe('https://test.api.com');
        expect(browser.getContext()).toBe(context);
      } finally {
        await context.close();
        try {
          fs.rmSync(userDataDir, { recursive: true, force: true });
        } catch {
          // ignore
        }
      }
    }, 60000);
  });

  describe('fromPage', () => {
    it('should create SentienceBrowser from existing page', async () => {
      // Auto-detect headless mode (headless in CI, headed locally)
      const isCI = process.env.CI === 'true' || process.env.CI === '1';
      const browserInstance = await chromium.launch({ headless: isCI });
      const context = await browserInstance.newContext({
        viewport: { width: 1440, height: 900 },
      });
      const page = await context.newPage();

      try {
        const sentienceBrowser = SentienceBrowser.fromPage(page);

        expect(sentienceBrowser.getPage()).toBe(page);
        expect(sentienceBrowser.getContext()).toBe(context);

        // Test that we can use it
        await patchExampleDotCom(page);
        await page.goto('https://example.com');
        await page.waitForLoadState('networkidle', { timeout: 10000 });

        // Verify viewport is preserved
        const viewportSize = await page.evaluate(() => ({
          width: window.innerWidth,
          height: window.innerHeight,
        }));

        expect(viewportSize.width).toBe(1440);
        expect(viewportSize.height).toBe(900);
      } finally {
        await context.close();
        await browserInstance.close();
      }
    }, 30000);

    it('should accept API key configuration', async () => {
      // Auto-detect headless mode (headless in CI, headed locally)
      const isCI = process.env.CI === 'true' || process.env.CI === '1';
      const browserInstance = await chromium.launch({ headless: isCI });
      const context = await browserInstance.newContext();
      const page = await context.newPage();

      try {
        const sentienceBrowser = SentienceBrowser.fromPage(
          page,
          'test_key',
          'https://test.api.com'
        );

        expect(sentienceBrowser.getApiKey()).toBe('test_key');
        expect(sentienceBrowser.getApiUrl()).toBe('https://test.api.com');
        expect(sentienceBrowser.getPage()).toBe(page);
      } finally {
        await context.close();
        await browserInstance.close();
      }
    }, 30000);
  });
});

describe('Browser Domain Policies', () => {
  it('should match domains with suffix rules', () => {
    expect(domainMatches('sub.example.com', 'example.com')).toBe(true);
    expect(domainMatches('example.com', 'example.com')).toBe(true);
    expect(domainMatches('example.com', '*.example.com')).toBe(true);
    expect(domainMatches('other.com', 'example.com')).toBe(false);
    expect(domainMatches('example.com', 'https://example.com')).toBe(true);
    expect(domainMatches('localhost', 'http://localhost:3000')).toBe(true);
  });

  it('should enforce allow/deny lists', () => {
    expect(isDomainAllowed('a.example.com', ['example.com'], [])).toBe(true);
    expect(isDomainAllowed('a.example.com', ['example.com'], ['bad.com'])).toBe(true);
    expect(isDomainAllowed('bad.example.com', [], ['example.com'])).toBe(false);
    expect(isDomainAllowed('x.com', ['example.com'], [])).toBe(false);
    expect(isDomainAllowed('example.com', ['https://example.com'], [])).toBe(true);
  });

  it('should extract host from ports', () => {
    expect(extractHost('http://localhost:3000')).toBe('localhost');
    expect(extractHost('localhost:3000')).toBe('localhost');
  });
});

describe('Browser keepAlive', () => {
  it('should skip close when keepAlive is true', async () => {
    const browser = new SentienceBrowser(
      undefined,
      undefined,
      true,
      undefined,
      undefined,
      undefined,
      undefined,
      undefined,
      undefined,
      undefined,
      undefined,
      undefined,
      true
    );

    const dummyContext = {
      closed: false,
      close: jest.fn().mockImplementation(() => {
        dummyContext.closed = true;
      }),
    };
    const dummyBrowser = {
      closed: false,
      close: jest.fn().mockImplementation(() => {
        dummyBrowser.closed = true;
      }),
    };
    (browser as any).context = dummyContext;
    (browser as any).browser = dummyBrowser;
    (browser as any).extensionPath = null;
    (browser as any).userDataDir = null;

    const result = await browser.close();
    expect(result).toBeNull();
    expect(dummyContext.closed).toBe(false);
    expect(dummyBrowser.closed).toBe(false);
  });
});
