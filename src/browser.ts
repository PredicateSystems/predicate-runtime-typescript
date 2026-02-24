/**
 * Playwright browser harness with extension loading
 */

import { chromium, BrowserContext, Page, Browser } from 'playwright';
import * as path from 'path';
import * as fs from 'fs';
import * as os from 'os';
import { URL } from 'url';
import { StorageState, Snapshot } from './types';
import { SnapshotOptions } from './snapshot';
import { IBrowser } from './protocols/browser-protocol';
import { snapshot as snapshotFunction } from './snapshot';

export type PermissionDefault = 'clear' | 'deny' | 'grant';

export type PermissionPolicy = {
  default?: PermissionDefault;
  autoGrant?: string[];
  geolocation?: { latitude: number; longitude: number; accuracy?: number };
  origin?: string;
};

export function normalizeDomain(domain: string): string {
  const raw = domain.trim();
  let host = raw;
  if (raw.includes('://')) {
    try {
      host = new URL(raw).hostname || '';
    } catch {
      host = raw;
    }
  } else {
    host = raw.split('/', 1)[0];
  }
  host = host.split(':', 1)[0];
  return host.trim().toLowerCase().replace(/^\./, '');
}

export function domainMatches(host: string, pattern: string): boolean {
  const hostNorm = normalizeDomain(host);
  let pat = normalizeDomain(pattern);
  if (pat.startsWith('*.')) {
    pat = pat.slice(2);
  }
  return hostNorm === pat || hostNorm.endsWith(`.${pat}`);
}

export function extractHost(url: string): string | null {
  let raw = url.trim();
  if (!raw.includes('://')) {
    raw = `https://${raw}`;
  }
  try {
    const parsed = new URL(raw);
    return parsed.hostname || null;
  } catch {
    return null;
  }
}

export function isDomainAllowed(
  host: string | null,
  allowed?: string[],
  prohibited?: string[]
): boolean {
  if (!host) return false;
  if (prohibited && prohibited.length > 0) {
    for (const pattern of prohibited) {
      if (domainMatches(host, pattern)) {
        return false;
      }
    }
  }
  if (allowed && allowed.length > 0) {
    return allowed.some(pattern => domainMatches(host, pattern));
  }
  return true;
}

export class SentienceBrowser implements IBrowser {
  private context: BrowserContext | null = null;
  private browser: Browser | null = null;
  private page: Page | null = null;
  private extensionPath: string | null = null;
  private userDataDir: string | null = null;
  private _apiKey?: string;
  private _apiUrl?: string;
  private headless: boolean;
  private _proxy?: string;
  private _userDataDir?: string;
  private _storageState?: string | StorageState | object;
  private _recordVideoDir?: string;
  private _recordVideoSize?: { width: number; height: number };
  private _viewport?: { width: number; height: number };
  private _deviceScaleFactor?: number;
  private _allowedDomains?: string[];
  private _prohibitedDomains?: string[];
  private _keepAlive: boolean;
  private _permissionPolicy?: PermissionPolicy;

  /**
   * Create a new SentienceBrowser instance
   *
   * @param apiKey - Optional API key for server-side processing (Pro/Enterprise tiers)
   * @param apiUrl - Optional API URL (defaults to https://api.sentienceapi.com if apiKey provided)
   * @param headless - Whether to run in headless mode (defaults to true in CI, false locally)
   * @param proxy - Optional proxy server URL (e.g., 'http://user:pass@proxy.example.com:8080')
   * @param userDataDir - Optional path to user data directory for persistent sessions
   * @param storageState - Optional storage state to inject (cookies + localStorage)
   * @param recordVideoDir - Optional directory path to save video recordings
   * @param recordVideoSize - Optional video resolution as object with 'width' and 'height' keys
   * @param viewport - Optional viewport size as object with 'width' and 'height' keys
   * @param deviceScaleFactor - Optional device scale factor to emulate high-DPI (Retina) screens.
   *                          Examples: 1.0 (default, standard DPI), 2.0 (Retina/high-DPI, like MacBook Pro), 3.0 (very high DPI)
   *                          If undefined, defaults to 1.0 (standard DPI).
   * @param allowedDomains - Optional list of allowed domains for navigation.
   * @param prohibitedDomains - Optional list of prohibited domains for navigation.
   * @param keepAlive - Keep browser alive after close() (no teardown).
   */
  constructor(
    apiKey?: string,
    apiUrl?: string,
    headless?: boolean,
    proxy?: string,
    userDataDir?: string,
    storageState?: string | StorageState | object,
    recordVideoDir?: string,
    recordVideoSize?: { width: number; height: number },
    viewport?: { width: number; height: number },
    deviceScaleFactor?: number,
    allowedDomains?: string[],
    prohibitedDomains?: string[],
    keepAlive: boolean = false,
    permissionPolicy?: PermissionPolicy
  ) {
    this._apiKey = apiKey;

    // Determine headless mode
    if (headless === undefined) {
      // Default to true in CI, false locally
      const ci = process.env.CI?.toLowerCase();
      this.headless = ci === 'true' || ci === '1' || ci === 'yes';
    } else {
      this.headless = headless;
    }

    // Configure API URL
    if (apiKey) {
      this._apiUrl = apiUrl || 'https://api.sentienceapi.com';
    } else {
      this._apiUrl = undefined;
    }

    // Support proxy from parameter or environment variable
    // Only use env var if it's a valid non-empty string
    const envProxy = process.env.SENTIENCE_PROXY;
    this._proxy = proxy || (envProxy && envProxy.trim() ? envProxy : undefined);

    // Auth injection support
    this._userDataDir = userDataDir;
    this._storageState = storageState;

    // Video recording support
    this._recordVideoDir = recordVideoDir;
    this._recordVideoSize = recordVideoSize || { width: 1280, height: 800 };

    // Viewport configuration
    this._viewport = viewport || { width: 1280, height: 800 };

    // Device scale factor for high-DPI emulation
    this._deviceScaleFactor = deviceScaleFactor;
    this._allowedDomains = allowedDomains;
    this._prohibitedDomains = prohibitedDomains;
    this._keepAlive = keepAlive;
    this._permissionPolicy = permissionPolicy;
  }

  private async applyPermissionPolicy(
    context: BrowserContext,
    policy: PermissionPolicy
  ): Promise<void> {
    const defaultPolicy = policy.default ?? 'clear';
    if (defaultPolicy === 'clear' || defaultPolicy === 'deny') {
      await context.clearPermissions();
    }
    if (policy.geolocation) {
      await context.setGeolocation(policy.geolocation);
    }
    if (policy.autoGrant && policy.autoGrant.length > 0) {
      const options = policy.origin ? { origin: policy.origin } : undefined;
      await context.grantPermissions(policy.autoGrant, options);
    }
  }

  async start(): Promise<void> {
    // 1. Resolve Extension Path
    // Handle: src/extension (local dev), dist/extension (prod), or ../sentience-chrome (monorepo)
    let extensionSource = '';

    const candidates = [
      // Production / Installed Package (npm)
      path.resolve(__dirname, '../extension'),
      path.resolve(__dirname, 'extension'),
      // Source directory (when running from src/ or npm package with src/)
      path.resolve(__dirname, '../src/extension'),
      path.resolve(__dirname, '../../src/extension'),
      // Local Monorepo Dev
      path.resolve(__dirname, '../../sentience-chrome'),
      path.resolve(__dirname, '../../../sentience-chrome'),
      // CI Artifact
      path.resolve(process.cwd(), 'extension'),
    ];

    for (const loc of candidates) {
      if (fs.existsSync(path.join(loc, 'manifest.json'))) {
        extensionSource = loc;
        break;
      }
    }

    if (!extensionSource) {
      throw new Error(
        `Sentience extension not found. Checked:\n${candidates.map(c => `- ${c}`).join('\n')}\n` +
          'Ensure the extension is built/downloaded.'
      );
    }

    // 2. Setup User Data Directory
    if (this._userDataDir) {
      // Use provided directory for persistent sessions
      this.userDataDir = this._userDataDir;
      if (!fs.existsSync(this.userDataDir)) {
        fs.mkdirSync(this.userDataDir, { recursive: true });
      }
    } else {
      // Create temp directory (ephemeral, existing behavior)
      this.userDataDir = fs.mkdtempSync(path.join(os.tmpdir(), 'sentience-ts-'));
    }

    this.extensionPath = path.join(this.userDataDir, 'extension');

    // Copy extension to temp dir
    this._copyRecursive(extensionSource, this.extensionPath);

    // 3. Build Args
    const args = [
      `--disable-extensions-except=${this.extensionPath}`,
      `--load-extension=${this.extensionPath}`,
      '--disable-blink-features=AutomationControlled',
      '--no-sandbox',
      '--disable-infobars',
    ];

    // CRITICAL: Headless Extensions Support
    // headless: true -> NO extensions.
    // headless: false + args: '--headless=new' -> YES extensions.
    if (this.headless) {
      args.push('--headless=new');
    }

    // CRITICAL: WebRTC leak protection for datacenter usage with proxies
    // Prevents WebRTC from leaking the real IP address even when using proxies
    if (this._proxy) {
      args.push('--disable-features=WebRtcHideLocalIpsWithMdns');
      args.push('--force-webrtc-ip-handling-policy=disable_non_proxied_udp');
    }

    // 4. Parse proxy configuration
    const proxyConfig = this.parseProxy(this._proxy);

    // 5. Setup video recording directory if requested
    if (this._recordVideoDir) {
      if (!fs.existsSync(this._recordVideoDir)) {
        fs.mkdirSync(this._recordVideoDir, { recursive: true });
      }
      console.log(`🎥 [Sentience] Recording video to: ${this._recordVideoDir}`);
      console.log(
        `   Resolution: ${this._recordVideoSize!.width}x${this._recordVideoSize!.height}`
      );
    }

    // 6. Launch Browser
    const launchOptions: any = {
      headless: false, // Must be false here, handled via args above
      args: args,
      viewport: this._viewport,
      // Clean User-Agent to avoid "HeadlessChrome" detection
      userAgent:
        'Mozilla/5.0 (Macintosh; Intel Mac OS X 10_15_7) AppleWebKit/537.36 (KHTML, like Gecko) Chrome/122.0.0.0 Safari/537.36',
      proxy: proxyConfig, // Pass proxy configuration
      // CRITICAL: Ignore HTTPS errors when using proxy (proxies often use self-signed certs)
      ignoreHTTPSErrors: proxyConfig !== undefined,
    };

    // Add device scale factor if configured
    if (this._deviceScaleFactor !== undefined) {
      launchOptions.deviceScaleFactor = this._deviceScaleFactor;
    }

    // Add video recording if configured
    if (this._recordVideoDir) {
      launchOptions.recordVideo = {
        dir: this._recordVideoDir,
        size: this._recordVideoSize,
      };
    }

    this.context = await chromium.launchPersistentContext(this.userDataDir, launchOptions);

    if (this._permissionPolicy) {
      await this.applyPermissionPolicy(this.context, this._permissionPolicy);
    }

    this.page = this.context.pages()[0] || (await this.context.newPage());

    // Inject storage state if provided (must be after context creation)
    if (this._storageState) {
      await this.injectStorageState(this._storageState);
    }

    // Apply context-level stealth patches (runs on every new page)
    await this.context.addInitScript(() => {
      // Early webdriver hiding - runs before any page script
      // Use multiple strategies to completely hide webdriver

      // Strategy 1: Try to delete it first
      try {
        delete (navigator as any).webdriver;
      } catch {
        // Property might not be deletable
      }

      // Strategy 2: Redefine to return undefined and hide from enumeration
      Object.defineProperty(navigator, 'webdriver', {
        get: () => undefined,
        configurable: true,
        enumerable: false,
        writable: false,
      });

      // Strategy 3: Override 'in' operator check
      const originalHasOwnProperty = Object.prototype.hasOwnProperty;
      Object.prototype.hasOwnProperty = function (prop: string | number | symbol) {
        if (this === navigator && (prop === 'webdriver' || prop === 'Webdriver')) {
          return false;
        }
        return originalHasOwnProperty.call(this, prop);
      };
    });

    // 5. Apply Comprehensive Stealth Patches
    // Use both CDP (earlier) and addInitScript (backup) for maximum coverage

    // Strategy A: Use CDP to inject at the earliest possible moment
    const client = await this.page.context().newCDPSession(this.page);
    await client.send('Page.addScriptToEvaluateOnNewDocument', {
      source: `
        // Aggressive webdriver hiding - must run before ANY page script
        Object.defineProperty(navigator, 'webdriver', {
          get: () => undefined,
          configurable: true,
          enumerable: false
        });
        
        // Override Object.getOwnPropertyDescriptor
        const originalGetOwnPropertyDescriptor = Object.getOwnPropertyDescriptor;
        Object.getOwnPropertyDescriptor = function(obj, prop) {
          if (obj === navigator && (prop === 'webdriver' || prop === 'Webdriver')) {
            return undefined;
          }
          return originalGetOwnPropertyDescriptor(obj, prop);
        };
        
        // Override Object.keys
        const originalKeys = Object.keys;
        Object.keys = function(obj) {
          const keys = originalKeys(obj);
          if (obj === navigator) {
            return keys.filter(k => k !== 'webdriver' && k !== 'Webdriver');
          }
          return keys;
        };
        
        // Override Object.getOwnPropertyNames
        const originalGetOwnPropertyNames = Object.getOwnPropertyNames;
        Object.getOwnPropertyNames = function(obj) {
          const names = originalGetOwnPropertyNames(obj);
          if (obj === navigator) {
            return names.filter(n => n !== 'webdriver' && n !== 'Webdriver');
          }
          return names;
        };
        
        // Override 'in' operator check
        const originalHasOwnProperty = Object.prototype.hasOwnProperty;
        Object.prototype.hasOwnProperty = function(prop) {
          if (this === navigator && (prop === 'webdriver' || prop === 'Webdriver')) {
            return false;
          }
          return originalHasOwnProperty.call(this, prop);
        };
      `,
    });

    // Strategy B: Also use addInitScript as backup (runs after CDP but before page scripts)
    await this.page.addInitScript(() => {
      // 1. Hide navigator.webdriver (comprehensive approach for advanced detection)
      // Advanced detection checks for property descriptor, so we need multiple strategies
      try {
        // Strategy 1: Try to delete the property
        delete (navigator as any).webdriver;
      } catch {
        // Property might not be deletable, continue with redefine
      }

      // Strategy 2: Redefine to return undefined (better than false)
      // Also set enumerable: false to hide from Object.keys() checks
      Object.defineProperty(navigator, 'webdriver', {
        get: () => undefined,
        configurable: true,
        enumerable: false,
      });

      // Strategy 3: Override Object.getOwnPropertyDescriptor only for navigator.webdriver
      // This prevents advanced detection that checks the property descriptor
      const originalGetOwnPropertyDescriptor = Object.getOwnPropertyDescriptor;
      Object.getOwnPropertyDescriptor = function (obj: any, prop: string | symbol) {
        if (obj === navigator && (prop === 'webdriver' || prop === 'Webdriver')) {
          return undefined;
        }
        return originalGetOwnPropertyDescriptor(obj, prop);
      } as any;

      // Strategy 4: Hide from Object.keys() and Object.getOwnPropertyNames()
      const originalKeys = Object.keys;
      Object.keys = function (obj: any) {
        const keys = originalKeys(obj);
        if (obj === navigator) {
          return keys.filter(k => k !== 'webdriver' && k !== 'Webdriver');
        }
        return keys;
      } as any;

      // Strategy 5: Hide from Object.getOwnPropertyNames()
      const originalGetOwnPropertyNames = Object.getOwnPropertyNames;
      Object.getOwnPropertyNames = function (obj: any) {
        const names = originalGetOwnPropertyNames(obj);
        if (obj === navigator) {
          return names.filter(n => n !== 'webdriver' && n !== 'Webdriver');
        }
        return names;
      } as any;

      // Strategy 6: Override hasOwnProperty to hide from 'in' operator checks
      const originalHasOwnProperty = Object.prototype.hasOwnProperty;
      Object.prototype.hasOwnProperty = function (prop: string | number | symbol) {
        if (this === navigator && (prop === 'webdriver' || prop === 'Webdriver')) {
          return false;
        }
        return originalHasOwnProperty.call(this, prop);
      };

      // 2. Inject window.chrome object (required for Chrome detection)
      if (typeof (window as any).chrome === 'undefined') {
        (window as any).chrome = {
          runtime: {},
          loadTimes: function () {},
          csi: function () {},
          app: {},
        };
      }

      // 3. Patch navigator.plugins (should have length > 0)
      // Only patch if plugins array is empty (headless mode issue)
      const originalPlugins = navigator.plugins;
      if (originalPlugins.length === 0) {
        // Create a PluginArray-like object with minimal plugins
        const fakePlugins = [
          {
            name: 'Chrome PDF Plugin',
            filename: 'internal-pdf-viewer',
            description: 'Portable Document Format',
            length: 1,
            item: function () {
              return null;
            },
            namedItem: function () {
              return null;
            },
          },
          {
            name: 'Chrome PDF Viewer',
            filename: 'mhjfbmdgcfjbbpaeojofohoefgiehjai',
            description: '',
            length: 0,
            item: function () {
              return null;
            },
            namedItem: function () {
              return null;
            },
          },
          {
            name: 'Native Client',
            filename: 'internal-nacl-plugin',
            description: '',
            length: 0,
            item: function () {
              return null;
            },
            namedItem: function () {
              return null;
            },
          },
        ];

        // Create PluginArray-like object (array-like but not a real array)
        // This needs to behave like the real PluginArray for detection to pass
        const pluginArray: any = {};
        fakePlugins.forEach((plugin, index) => {
          Object.defineProperty(pluginArray, index.toString(), {
            value: plugin,
            enumerable: true,
            configurable: true,
          });
        });

        Object.defineProperty(pluginArray, 'length', {
          value: fakePlugins.length,
          enumerable: false,
          configurable: false,
        });

        pluginArray.item = function (index: number) {
          return this[index] || null;
        };
        pluginArray.namedItem = function (name: string) {
          for (let i = 0; i < this.length; i++) {
            if (this[i] && this[i].name === name) return this[i];
          }
          return null;
        };

        // Make it iterable (for for...of loops)
        pluginArray[Symbol.iterator] = function* () {
          for (let i = 0; i < this.length; i++) {
            yield this[i];
          }
        };

        // Make it array-like for Array.from() and spread
        Object.setPrototypeOf(pluginArray, Object.create(null));

        Object.defineProperty(navigator, 'plugins', {
          get: () => pluginArray,
          configurable: true,
          enumerable: true,
        });
      }

      // 4. Ensure navigator.languages exists and has values
      if (!navigator.languages || navigator.languages.length === 0) {
        Object.defineProperty(navigator, 'languages', {
          get: () => ['en-US', 'en'],
          configurable: true,
        });
      }

      // 5. Patch permissions API (should exist)
      if (!navigator.permissions) {
        (navigator as any).permissions = {
          query: (_parameters: PermissionDescriptor) => {
            return { state: 'granted', onchange: null } as PermissionStatus;
          },
        };
      }
    });

    // Inject API Key if present
    if (this._apiKey) {
      await this.page.addInitScript(key => {
        (window as any).__SENTIENCE_API_KEY__ = key;
      }, this._apiKey);
    }

    // Wait for extension background pages to spin up
    await new Promise(r => setTimeout(r, 500));
  }

  async goto(url: string): Promise<void> {
    const page = this.getPage();
    if (!page) {
      throw new Error('Browser not started. Call start() first.');
    }
    const host = extractHost(url);
    if (!isDomainAllowed(host, this._allowedDomains, this._prohibitedDomains)) {
      throw new Error(`domain not allowed: ${host}`);
    }
    await page.goto(url, { waitUntil: 'domcontentloaded' });

    if (!(await this.waitForExtension(page, 15000))) {
      // Gather Debug Info
      const diag = await page
        .evaluate(() => ({
          sentience_global: typeof (window as any).sentience !== 'undefined',
          wasm_ready: (window as any).sentience && (window as any).sentience._wasmModule !== null,
          ext_id: document.documentElement.dataset.sentienceExtensionId || 'not set',
          url: window.location.href,
        }))
        .catch(e => ({ error: String(e) }));

      throw new Error(
        'Extension failed to load after navigation.\n' +
          `Path: ${this.extensionPath}\n` +
          `Diagnostics: ${JSON.stringify(diag, null, 2)}`
      );
    }
  }

  private async waitForExtension(page: Page, timeoutMs: number): Promise<boolean> {
    const start = Date.now();
    while (Date.now() - start < timeoutMs) {
      try {
        const ready = await page.evaluate(() => {
          // Check for API AND Wasm Module (set by injected_api.js)
          const s = (window as any).sentience;
          return s && s._wasmModule !== null; // Strict check for null (it's initialized as null)
        });
        if (ready) return true;
      } catch {
        // Context invalid errors expected during navigation
      }
      await new Promise(r => setTimeout(r, 100));
    }
    return false;
  }

  getPage(): Page | null {
    return this.page;
  }

  // Helper for recursive copy (fs.cp is Node 16.7+)
  private _copyRecursive(src: string, dest: string) {
    if (fs.statSync(src).isDirectory()) {
      if (!fs.existsSync(dest)) fs.mkdirSync(dest);
      fs.readdirSync(src).forEach(child => {
        this._copyRecursive(path.join(src, child), path.join(dest, child));
      });
    } else {
      fs.copyFileSync(src, dest);
    }
  }

  // Expose API configuration (read-only)
  getApiKey(): string | undefined {
    return this._apiKey;
  }

  getApiUrl(): string | undefined {
    return this._apiUrl;
  }

  /**
   * Take a snapshot of the current page
   * Implements IBrowser interface
   */
  async snapshot(options?: SnapshotOptions): Promise<Snapshot> {
    return snapshotFunction(this, options);
  }

  /**
   * Parse proxy connection string into Playwright format.
   *
   * @param proxyString - Standard format "http://username:password@host:port"
   *                    or "socks5://user:pass@host:port"
   * @returns Playwright proxy object or undefined if invalid
   */
  private parseProxy(
    proxyString?: string
  ): { server: string; username?: string; password?: string } | undefined {
    if (!proxyString || !proxyString.trim()) {
      return undefined;
    }

    try {
      const parsed = new URL(proxyString);

      // Validate scheme
      const validSchemes = ['http:', 'https:', 'socks5:'];
      if (!validSchemes.includes(parsed.protocol)) {
        throw new Error(`Unsupported proxy scheme: ${parsed.protocol}`);
      }

      // Validate host and port
      if (!parsed.hostname || !parsed.port) {
        throw new Error('Proxy URL must include hostname and port');
      }

      // Build Playwright proxy object
      const proxyConfig: { server: string; username?: string; password?: string } = {
        server: `${parsed.protocol}//${parsed.hostname}:${parsed.port}`,
      };

      // Add credentials if present
      if (parsed.username && parsed.password) {
        proxyConfig.username = parsed.username;
        proxyConfig.password = parsed.password;
      }

      return proxyConfig;
    } catch (e: any) {
      console.warn(`⚠️  [Sentience] Invalid proxy configuration: ${e.message}`);
      console.warn('   Expected format: http://username:password@host:port');
      return undefined;
    }
  }

  /**
   * Inject storage state (cookies + localStorage) into browser context.
   *
   * @param storageState - Path to JSON file, StorageState object, or plain object
   */
  private async injectStorageState(storageState: string | StorageState | object): Promise<void> {
    // Load storage state
    let state: StorageState;

    if (typeof storageState === 'string') {
      // Load from file
      const content = fs.readFileSync(storageState, 'utf-8');
      state = JSON.parse(content) as StorageState;
    } else if (typeof storageState === 'object' && storageState !== null) {
      // Already an object (StorageState or plain object)
      state = storageState as StorageState;
    } else {
      throw new Error(
        `Invalid storageState type: ${typeof storageState}. ` +
          'Expected string (file path), StorageState, or object.'
      );
    }

    // Inject cookies (works globally)
    if (state.cookies && Array.isArray(state.cookies) && state.cookies.length > 0) {
      // Convert to Playwright cookie format
      const playwrightCookies = state.cookies.map(cookie => {
        const playwrightCookie: any = {
          name: cookie.name,
          value: cookie.value,
          domain: cookie.domain,
          path: cookie.path || '/',
        };

        if (cookie.expires !== undefined) {
          playwrightCookie.expires = cookie.expires;
        }
        if (cookie.httpOnly !== undefined) {
          playwrightCookie.httpOnly = cookie.httpOnly;
        }
        if (cookie.secure !== undefined) {
          playwrightCookie.secure = cookie.secure;
        }
        if (cookie.sameSite !== undefined) {
          playwrightCookie.sameSite = cookie.sameSite;
        }

        return playwrightCookie;
      });

      await this.context!.addCookies(playwrightCookies);
      console.log(`✅ [Sentience] Injected ${state.cookies.length} cookie(s)`);
    }

    // Inject LocalStorage (requires navigation to each domain)
    if (state.origins && Array.isArray(state.origins)) {
      for (const originData of state.origins) {
        const origin = originData.origin;
        if (!origin) {
          continue;
        }

        try {
          // Navigate to origin
          await this.page!.goto(origin, { waitUntil: 'domcontentloaded', timeout: 10000 });

          // Inject localStorage
          if (originData.localStorage && Array.isArray(originData.localStorage)) {
            // Convert to dict format for JavaScript
            const localStorageDict: Record<string, string> = {};
            for (const item of originData.localStorage) {
              localStorageDict[item.name] = item.value;
            }

            await this.page!.evaluate((localStorageData: Record<string, string>) => {
              for (const [key, value] of Object.entries(localStorageData)) {
                localStorage.setItem(key, value);
              }
            }, localStorageDict);

            console.log(
              `✅ [Sentience] Injected ${originData.localStorage.length} localStorage item(s) for ${origin}`
            );
          }
        } catch (error: any) {
          console.warn(
            `⚠️  [Sentience] Failed to inject localStorage for ${origin}: ${error.message}`
          );
        }
      }
    }
  }

  /**
   * Get the browser context (for utilities like saveStorageState)
   */
  getContext(): BrowserContext | null {
    return this.context;
  }

  /**
   * Create SentienceBrowser from an existing Playwright BrowserContext.
   *
   * This allows you to use Sentience SDK with a browser context you've already created,
   * giving you more control over browser initialization.
   *
   * @param context - Existing Playwright BrowserContext
   * @param apiKey - Optional API key for server-side processing
   * @param apiUrl - Optional API URL (defaults to https://api.sentienceapi.com if apiKey provided)
   * @returns SentienceBrowser instance configured to use the existing context
   *
   * @example
   * ```typescript
   * import { chromium } from 'playwright';
   * import { SentienceBrowser } from '@sentience/sdk';
   *
   * const context = await chromium.launchPersistentContext(...);
   * const browser = SentienceBrowser.fromExisting(context);
   * await browser.getPage().goto('https://example.com');
   * ```
   */
  static async fromExisting(
    context: BrowserContext,
    apiKey?: string,
    apiUrl?: string
  ): Promise<SentienceBrowser> {
    const instance = new SentienceBrowser(apiKey, apiUrl);
    instance.context = context;
    const pages = context.pages();
    instance.page = pages.length > 0 ? pages[0] : await context.newPage();

    // Wait for extension to be ready (if extension is loaded)
    // Note: In TypeScript, we can't easily apply stealth here without the page
    // The user should ensure stealth is applied to their context if needed

    return instance;
  }

  /**
   * Create SentienceBrowser from an existing Playwright Page.
   *
   * This allows you to use Sentience SDK with a page you've already created,
   * giving you more control over browser initialization.
   *
   * @param page - Existing Playwright Page
   * @param apiKey - Optional API key for server-side processing
   * @param apiUrl - Optional API URL (defaults to https://api.sentienceapi.com if apiKey provided)
   * @returns SentienceBrowser instance configured to use the existing page
   *
   * @example
   * ```typescript
   * import { chromium } from 'playwright';
   * import { SentienceBrowser } from '@sentience/sdk';
   *
   * const browserInstance = await chromium.launch();
   * const context = await browserInstance.newContext();
   * const page = await context.newPage();
   * await page.goto('https://example.com');
   *
   * const browser = SentienceBrowser.fromPage(page);
   * ```
   */
  static fromPage(page: Page, apiKey?: string, apiUrl?: string): SentienceBrowser {
    const instance = new SentienceBrowser(apiKey, apiUrl);
    instance.page = page;
    instance.context = page.context();

    // Wait for extension to be ready (if extension is loaded)
    // Note: In TypeScript, we can't easily apply stealth here without the page
    // The user should ensure stealth is applied to their context if needed

    return instance;
  }

  async close(outputPath?: string): Promise<string | null> {
    if (this._keepAlive) {
      return null;
    }
    let tempVideoPath: string | null = null;

    // Get video path before closing (if recording was enabled)
    // Note: Playwright saves videos when pages/context close, but we can get the
    // expected path before closing. The actual file will be available after close.
    if (this._recordVideoDir) {
      try {
        // Try to get video path from the first page
        if (this.page) {
          const video = this.page.video();
          if (video) {
            tempVideoPath = await video.path();
          }
        }
        // If that fails, check all pages in the context (before closing)
        if (!tempVideoPath && this.context) {
          const pages = this.context.pages();
          for (const page of pages) {
            try {
              const video = page.video();
              if (video) {
                tempVideoPath = await video.path();
                break;
              }
            } catch {
              // Continue to next page
            }
          }
        }
      } catch {
        // Video path might not be available until after close
        // We'll use fallback mechanism below
      }
    }

    const cleanup: Promise<void>[] = [];

    // Close context first (this also closes the browser for persistent contexts)
    // This triggers video file finalization
    if (this.context) {
      cleanup.push(
        this.context.close().catch(() => {
          // Ignore errors during cleanup
        })
      );
      this.context = null;
    }

    // Close browser if it exists (for non-persistent contexts)
    if (this.browser) {
      cleanup.push(
        this.browser.close().catch(() => {
          // Ignore errors during cleanup
        })
      );
      this.browser = null;
    }

    // Wait for all cleanup to complete
    await Promise.all(cleanup);

    // Clean up extension directory
    if (this.extensionPath && fs.existsSync(this.extensionPath)) {
      try {
        fs.rmSync(this.extensionPath, { recursive: true, force: true });
      } catch {
        // Ignore cleanup errors
      }
      this.extensionPath = null;
    }

    // After context closes, verify video file exists if we have a path
    let finalPath = tempVideoPath;
    if (tempVideoPath && fs.existsSync(tempVideoPath)) {
      // Video file exists, proceed with rename if needed
    } else if (this._recordVideoDir && fs.existsSync(this._recordVideoDir)) {
      // Fallback: If we couldn't get the path but recording was enabled,
      // check the directory for video files
      try {
        const videoFiles = fs
          .readdirSync(this._recordVideoDir)
          .filter(f => f.endsWith('.webm'))
          .map(f => ({
            path: path.join(this._recordVideoDir!, f),
            mtime: fs.statSync(path.join(this._recordVideoDir!, f)).mtime.getTime(),
          }))
          .sort((a, b) => b.mtime - a.mtime); // Most recent first

        if (videoFiles.length > 0) {
          finalPath = videoFiles[0].path;
        }
      } catch {
        // Ignore errors when scanning directory
      }
    }

    // Rename/move video if output_path is specified
    if (finalPath && outputPath && fs.existsSync(finalPath)) {
      try {
        // Ensure parent directory exists
        const outputDir = path.dirname(outputPath);
        if (!fs.existsSync(outputDir)) {
          fs.mkdirSync(outputDir, { recursive: true });
        }
        fs.renameSync(finalPath, outputPath);
        finalPath = outputPath;
      } catch (error: any) {
        console.warn(`Failed to rename video file: ${error.message}`);
        // Return original path if rename fails
      }
    }

    // Clean up user data directory (only if it's a temp directory)
    // If user provided a custom userDataDir, we don't delete it (persistent sessions)
    if (this.userDataDir && fs.existsSync(this.userDataDir)) {
      // Only delete if it's a temp directory (starts with os.tmpdir())
      const isTempDir = this.userDataDir.startsWith(os.tmpdir());
      if (isTempDir) {
        try {
          fs.rmSync(this.userDataDir, { recursive: true, force: true });
        } catch {
          // Ignore cleanup errors
        }
      }
      this.userDataDir = null;
    }

    return finalPath;
  }
}

/**
 * Predicate rebrand alias for SentienceBrowser.
 * Kept as a runtime alias to avoid breaking existing integrations.
 */
export const PredicateBrowser = SentienceBrowser;
