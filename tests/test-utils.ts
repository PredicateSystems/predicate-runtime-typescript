/**
 * Test utilities for browser tests
 */

import { SentienceBrowser } from '../src';
import { Page } from 'playwright';

/**
 * Creates a browser instance and starts it with better error handling
 * Auto-detects headless mode based on CI environment (headless in CI, headed locally)
 */
export async function createTestBrowser(headless?: boolean): Promise<SentienceBrowser> {
  const browser = new SentienceBrowser(undefined, undefined, headless);
  try {
    await browser.start();
    const page = browser.getPage();
    if (page) {
      await patchExampleDotCom(page);
    }
    return browser;
  } catch (e: any) {
    // Clean up browser on failure to prevent resource leaks
    try {
      await browser.close();
    } catch (closeError) {
      // Ignore cleanup errors
    }
    // Enhance error message but don't log here (Jest will handle it)
    const enhancedError = new Error(
      `Browser startup failed: ${e.message}\n` +
        'Make sure:\n' +
        '1. Playwright browsers are installed: npx playwright install chromium\n' +
        '2. Extension is built: cd sentience-chrome && ./build.sh'
    );
    enhancedError.stack = e.stack;
    throw enhancedError;
  }
}

/**
 * Gets the page from browser and throws if it's null
 * Helper function for tests to avoid repetitive null checks
 */
export function getPageOrThrow(browser: SentienceBrowser): Page {
  const page = browser.getPage();
  if (!page) {
    throw new Error('Browser page is not available. Make sure browser.start() was called.');
  }
  return page;
}

const DEFAULT_TEST_HTML = `<!doctype html>
<html>
  <head><meta charset="utf-8" /></head>
  <body>
    <a id="link" href="#ok">Example Link</a>
    <input id="text" type="text" value="hello" />
    <button id="btn" type="button">Click me</button>
    <div style="height: 2000px;"></div>
  </body>
</html>`;

export async function setTestPageContent(page: Page, html?: string): Promise<void> {
  await page.setContent(html ?? DEFAULT_TEST_HTML, { waitUntil: 'domcontentloaded' });
}

export async function patchExampleDotCom(page: Page): Promise<void> {
  await page.route(/https?:\/\/example\.com\/?.*/, async route => {
    await route.fulfill({
      status: 200,
      contentType: 'text/html',
      body: DEFAULT_TEST_HTML,
    });
  });
}

const SEARCH_RESULTS_HTML = `<!doctype html>
<html>
  <head><meta charset="utf-8" /></head>
  <body>
    <form role="search">
      <input aria-label="Search" type="search" value="" />
      <button type="submit">Search</button>
    </form>
    <main>
      <a href="https://example.com/result">Sentience SDK Result</a>
    </main>
  </body>
</html>`;

export async function patchSearchEnginePages(page: Page): Promise<void> {
  await page.route(
    /https?:\/\/(duckduckgo\.com|www\.google\.com|www\.bing\.com)\/.*/,
    async route => {
      await route.fulfill({
        status: 200,
        contentType: 'text/html',
        body: SEARCH_RESULTS_HTML,
      });
    }
  );
}
