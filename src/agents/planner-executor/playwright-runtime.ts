/**
 * PlaywrightRuntime: Browser automation using Playwright/Chromium.
 *
 * Implements the AgentRuntime interface for PlannerExecutorAgent using
 * Playwright for real browser automation with the Predicate browser extension.
 */

import { Page, BrowserContext } from 'playwright';
import { PredicateBrowser } from '../../browser';
import { snapshot as takeSnapshot } from '../../snapshot';
import { click, typeText, press } from '../../actions';
import type { Snapshot as SDKSnapshot, Element as SDKElement } from '../../types';
import type { Snapshot, SnapshotElement } from './plan-models';
import type { AgentRuntime } from './planner-executor-agent';

/**
 * Options for creating a PlaywrightRuntime.
 */
export interface PlaywrightRuntimeOptions {
  /**
   * Run browser in headless mode.
   * Defaults to true in CI, false locally.
   */
  headless?: boolean;

  /**
   * API key for Predicate/Sentience backend processing.
   */
  apiKey?: string;

  /**
   * API URL for Predicate/Sentience backend.
   */
  apiUrl?: string;

  /**
   * Proxy server URL (e.g., 'http://user:pass@proxy.example.com:8080').
   */
  proxy?: string;

  /**
   * Path to user data directory for persistent sessions.
   */
  userDataDir?: string;

  /**
   * Storage state to inject (cookies + localStorage).
   */
  storageState?: string | object;

  /**
   * Directory to save video recordings.
   */
  recordVideoDir?: string;

  /**
   * Video resolution.
   */
  recordVideoSize?: { width: number; height: number };

  /**
   * Viewport size.
   */
  viewport?: { width: number; height: number };

  /**
   * Device scale factor (e.g., 2.0 for Retina).
   */
  deviceScaleFactor?: number;

  /**
   * Allowed domains for navigation.
   */
  allowedDomains?: string[];

  /**
   * Prohibited domains for navigation.
   */
  prohibitedDomains?: string[];

  /**
   * Keep browser alive after close() (no teardown).
   */
  keepAlive?: boolean;

  /**
   * Default timeout for operations (ms).
   */
  timeout?: number;

  /**
   * Show visual overlay highlighting elements in browser.
   * Useful for debugging and demos.
   */
  showOverlay?: boolean;
}

/**
 * PlaywrightRuntime implements AgentRuntime using Playwright/Chromium.
 *
 * Provides real browser automation with the Predicate browser extension for
 * snapshot-based element selection and interaction.
 *
 * @example
 * ```typescript
 * const runtime = new PlaywrightRuntime({ headless: false });
 * await runtime.start();
 *
 * const agent = new PlannerExecutorAgent({ planner, executor });
 * const result = await agent.runStepwise(runtime, {
 *   task: 'Search for laptops on Amazon',
 *   startUrl: 'https://www.amazon.com',
 * });
 *
 * await runtime.close();
 * ```
 */
export class PlaywrightRuntime implements AgentRuntime {
  private browser: InstanceType<typeof PredicateBrowser>;
  private options: PlaywrightRuntimeOptions;
  private started = false;

  constructor(options: PlaywrightRuntimeOptions = {}) {
    this.options = {
      timeout: 30000,
      ...options,
    };

    // Create PredicateBrowser with options
    this.browser = new PredicateBrowser(
      options.apiKey,
      options.apiUrl,
      options.headless,
      options.proxy,
      options.userDataDir,
      options.storageState,
      options.recordVideoDir,
      options.recordVideoSize,
      options.viewport,
      options.deviceScaleFactor,
      options.allowedDomains,
      options.prohibitedDomains,
      options.keepAlive ?? false
    );
  }

  /**
   * Start the browser and initialize the runtime.
   */
  async start(): Promise<void> {
    if (this.started) {
      return;
    }

    await this.browser.start();
    this.started = true;
  }

  /**
   * Close the browser and clean up resources.
   *
   * @param outputPath - Optional path to save video recording
   * @returns Path to video file if recording was enabled
   */
  async close(outputPath?: string): Promise<string | null> {
    if (!this.started) {
      return null;
    }

    const videoPath = await this.browser.close(outputPath);
    this.started = false;
    return videoPath;
  }

  /**
   * Get the underlying Playwright Page instance.
   */
  getPage(): Page | null {
    return this.browser.getPage();
  }

  /**
   * Get the underlying BrowserContext.
   */
  getContext(): BrowserContext | null {
    return this.browser.getContext();
  }

  /**
   * Ensure browser is started.
   */
  private ensureStarted(): void {
    if (!this.started) {
      throw new Error('PlaywrightRuntime not started. Call start() first.');
    }
  }

  // ---------------------------------------------------------------------------
  // AgentRuntime Interface Implementation
  // ---------------------------------------------------------------------------

  /**
   * Take a snapshot of the current page.
   *
   * Uses the Predicate browser extension for semantic element extraction with
   * importance ranking and pruning.
   */
  async snapshot(options?: {
    limit?: number;
    screenshot?: boolean;
    goal?: string;
  }): Promise<Snapshot | null> {
    this.ensureStarted();

    const page = this.browser.getPage();

    // Wait for page to be stable before taking snapshot
    if (page) {
      try {
        await page.waitForLoadState('domcontentloaded', { timeout: 5000 });
      } catch {
        // Best effort - continue even if timeout
      }
    }

    try {
      // Get snapshot using SDK's snapshot function
      const snap = await takeSnapshot(this.browser, {
        limit: options?.limit || 100,
        screenshot: options?.screenshot ?? false,
        goal: options?.goal,
        show_overlay: this.options.showOverlay ?? true,
      });

      if (!snap || snap.status !== 'success') {
        return null;
      }

      // Convert to AgentRuntime Snapshot format
      return await this.convertSnapshot(snap);
    } catch (e) {
      console.error('[PlaywrightRuntime] Snapshot error:', e);
      return null;
    }
  }

  /**
   * Convert SDK snapshot to AgentRuntime format.
   */
  private async convertSnapshot(snap: SDKSnapshot): Promise<Snapshot> {
    const page = this.browser.getPage();
    const title = page ? await page.title() : '';

    const elements = (snap.elements || []).map(el => this.convertElement(el));

    // Debug: Log element role distribution
    const roleCounts = new Map<string, number>();
    const clickableCounts = new Map<string, number>();
    for (const el of elements) {
      const role = (el.role || 'none').toLowerCase();
      roleCounts.set(role, (roleCounts.get(role) || 0) + 1);
      if (el.clickable) {
        clickableCounts.set(role, (clickableCounts.get(role) || 0) + 1);
      }
    }
    const rolesSummary = Array.from(roleCounts.entries())
      .sort((a, b) => b[1] - a[1])
      .slice(0, 6)
      .map(([r, c]) => `${r}:${c}`)
      .join(', ');
    const clickableSummary = Array.from(clickableCounts.entries())
      .sort((a, b) => b[1] - a[1])
      .map(([r, c]) => `${r}:${c}`)
      .join(', ');
    console.log(`  [convertSnapshot] Elements: ${elements.length}, Roles: ${rolesSummary}`);
    console.log(`  [convertSnapshot] Clickable by role: ${clickableSummary || 'none'}`);

    return {
      url: snap.url,
      title,
      elements,
      screenshot: snap.screenshot,
      status: snap.status,
    };
  }

  /**
   * Convert snapshot element to AgentRuntime format.
   */
  private convertElement(el: SDKElement): SnapshotElement {
    // SDK Element uses different field naming conventions
    // Convert to agent-friendly format
    // Extract clickable and isPrimary from visual_cues if available
    const visualCues = (el as any).visual_cues;

    // Determine clickable status:
    // 1. Input elements (textbox, searchbox, combobox, input, textarea) are ALWAYS interactive
    // 2. Elements with href are ALWAYS clickable (they're links)
    // 3. For other elements, use visual_cues.is_clickable or fall back to role check
    const role = (el.role || '').toLowerCase();
    const isInputRole = ['textbox', 'searchbox', 'combobox', 'input', 'textarea'].includes(role);
    const hasHref = Boolean(el.href);
    const isClickable =
      isInputRole || hasHref || (visualCues?.is_clickable ?? this.isInteractiveRole(el.role));

    const isPrimary = visualCues?.is_primary ?? false;
    const hasBackground = Boolean(visualCues?.background_color_name);

    return {
      id: el.id,
      role: el.role || '',
      text: el.text || el.name || '',
      name: el.name || undefined,
      clickable: isClickable,
      importance: el.importance ?? 0,
      isPrimary,
      background: hasBackground,
      nearbyText: el.nearby_text || undefined,
      ordinal: el.group_index?.toString(),
      inDominantGroup: el.group_key !== undefined,
      href: el.href,
    };
  }

  /**
   * Check if a role is typically interactive.
   */
  private isInteractiveRole(role: string): boolean {
    const interactiveRoles = new Set([
      'button',
      'link',
      'textbox',
      'searchbox',
      'combobox',
      'checkbox',
      'radio',
      'slider',
      'tab',
      'menuitem',
      'option',
      'switch',
      'a',
      'input',
      'select',
      'textarea',
    ]);
    return interactiveRoles.has((role || '').toLowerCase());
  }

  /**
   * Navigate to a URL.
   */
  async goto(url: string): Promise<void> {
    this.ensureStarted();
    await this.browser.goto(url);

    // Wait for page to be ready after navigation
    const page = this.browser.getPage();
    if (page) {
      try {
        await page.waitForLoadState('domcontentloaded', { timeout: 10000 });
        // Small delay to let any client-side JS settle
        await page.waitForTimeout(500);
      } catch {
        // Best effort - continue even if timeout
      }
    }
  }

  /**
   * Click an element by its snapshot ID.
   *
   * Uses the Predicate browser extension's element registry to find and click
   * the element by its semantic ID.
   */
  async click(elementId: number): Promise<void> {
    this.ensureStarted();

    const result = await click(this.browser, elementId);
    if (!result.success) {
      throw new Error(result.error?.reason || `Click failed for element ${elementId}`);
    }
  }

  /**
   * Type text into an element by its snapshot ID.
   */
  async type(elementId: number, text: string): Promise<void> {
    this.ensureStarted();

    const result = await typeText(this.browser, elementId, text);
    if (!result.success) {
      throw new Error(result.error?.reason || `Type failed for element ${elementId}`);
    }
  }

  /**
   * Press a keyboard key.
   */
  async pressKey(key: string): Promise<void> {
    this.ensureStarted();

    const result = await press(this.browser, key);
    if (!result.success) {
      throw new Error(result.error?.reason || `Press failed for key ${key}`);
    }
  }

  /**
   * Scroll the page in a direction.
   */
  async scroll(direction: 'up' | 'down'): Promise<void> {
    this.ensureStarted();

    const page = this.browser.getPage();
    if (!page) {
      throw new Error('Page not available');
    }

    const delta = direction === 'down' ? 400 : -400;
    await page.mouse.wheel(0, delta);
    // Wait for scroll to take effect
    await page.waitForTimeout(200);
  }

  /**
   * Get the current URL.
   */
  async getCurrentUrl(): Promise<string> {
    this.ensureStarted();

    const page = this.browser.getPage();
    if (!page) {
      throw new Error('Page not available');
    }

    return page.url();
  }

  /**
   * Read the current page content as structured markdown.
   * Uses the Sentience extension's read() method (convertToMarkdown) when available,
   * falling back to innerText extraction.
   */
  async readMarkdown(options?: { maxChars?: number }): Promise<string | null> {
    this.ensureStarted();

    const page = this.browser.getPage();
    if (!page) {
      throw new Error('Page not available');
    }

    const maxChars = options?.maxChars ?? 50000;

    try {
      const hasSentience = await page.evaluate(
        () => typeof (window as any).sentience?.read === 'function'
      );

      if (hasSentience) {
        const result = await page.evaluate((max: number) => {
          const readResult = (window as any).sentience.read({ format: 'markdown' });
          if (readResult?.status === 'success' && readResult.content) {
            let content = readResult.content;
            if (content.length > max) {
              content = content.slice(0, max) + '\n\n[... content truncated ...]';
            }
            return content;
          }
          return null;
        }, maxChars);

        if (result) return result;
      }
    } catch {
      // Sentience extension not available, fall through
    }

    try {
      const text = await page.locator('body').innerText({ timeout: 5000 });
      let result = text.replace(/\n{3,}/g, '\n\n').trim();
      if (result.length > maxChars) {
        result = result.slice(0, maxChars) + '\n\n[... content truncated ...]';
      }
      return result || null;
    } catch {
      const text = await page.evaluate(() => {
        const el = document.querySelector('body');
        return el?.textContent ?? '';
      });
      let result = text.replace(/\n{3,}/g, '\n\n').trim();
      if (result.length > maxChars) {
        result = result.slice(0, maxChars) + '\n\n[... content truncated ...]';
      }
      return result || null;
    }
  }

  /**
   * Get the viewport height.
   */
  async getViewportHeight(): Promise<number> {
    this.ensureStarted();

    const page = this.browser.getPage();
    if (!page) {
      throw new Error('Page not available');
    }

    const viewport = page.viewportSize();
    return viewport?.height || 800;
  }

  /**
   * Scroll by a delta amount.
   *
   * @returns true if scroll was successful
   */
  async scrollBy(dy: number): Promise<boolean> {
    this.ensureStarted();

    const page = this.browser.getPage();
    if (!page) {
      return false;
    }

    try {
      const beforeY = await page.evaluate(() => window.scrollY);
      await page.mouse.wheel(0, dy);
      // Wait for scroll to take effect
      await page.waitForTimeout(100);
      const afterY = await page.evaluate(() => window.scrollY);
      return Math.abs(afterY - beforeY) > 10;
    } catch {
      return false;
    }
  }

  // ---------------------------------------------------------------------------
  // Additional Utility Methods
  // ---------------------------------------------------------------------------

  /**
   * Wait for navigation to complete.
   */
  async waitForNavigation(options?: { timeout?: number }): Promise<void> {
    this.ensureStarted();

    const page = this.browser.getPage();
    if (!page) {
      throw new Error('Page not available');
    }

    await page.waitForLoadState('domcontentloaded', {
      timeout: options?.timeout || this.options.timeout,
    });
  }

  /**
   * Wait for an element to appear.
   */
  async waitForElement(selector: string, options?: { timeout?: number }): Promise<void> {
    this.ensureStarted();

    const page = this.browser.getPage();
    if (!page) {
      throw new Error('Page not available');
    }

    await page.waitForSelector(selector, {
      timeout: options?.timeout || this.options.timeout,
    });
  }

  /**
   * Take a screenshot.
   *
   * @param path - Optional path to save screenshot
   * @returns Screenshot as base64 string
   */
  async screenshot(path?: string): Promise<string> {
    this.ensureStarted();

    const page = this.browser.getPage();
    if (!page) {
      throw new Error('Page not available');
    }

    const buffer = await page.screenshot({
      path,
      type: 'jpeg',
      quality: 80,
    });
    return buffer.toString('base64');
  }

  /**
   * Execute JavaScript in the page context.
   */
  async evaluate<T>(code: string): Promise<T> {
    this.ensureStarted();

    const page = this.browser.getPage();
    if (!page) {
      throw new Error('Page not available');
    }

    return page.evaluate(code);
  }

  /**
   * Create PlaywrightRuntime from an existing Playwright Page.
   *
   * Useful for integrating with existing Playwright test setups.
   */
  static async fromPage(
    page: Page,
    options?: Partial<PlaywrightRuntimeOptions>
  ): Promise<PlaywrightRuntime> {
    const runtime = new PlaywrightRuntime(options);
    // Use PredicateBrowser.fromPage to wrap existing page
    (runtime as any).browser = PredicateBrowser.fromPage(page, options?.apiKey, options?.apiUrl);
    runtime.started = true;
    return runtime;
  }
}

/**
 * Create and start a PlaywrightRuntime.
 *
 * Convenience function that creates and starts the runtime in one call.
 *
 * @example
 * ```typescript
 * const runtime = await createPlaywrightRuntime({ headless: false });
 * // ... use runtime
 * await runtime.close();
 * ```
 */
export async function createPlaywrightRuntime(
  options?: PlaywrightRuntimeOptions
): Promise<PlaywrightRuntime> {
  const runtime = new PlaywrightRuntime(options);
  await runtime.start();
  return runtime;
}
