/**
 * Tests for actions (click, type, press, clickRect)
 */

import {
  SentienceBrowser,
  back,
  check,
  clear,
  click,
  search,
  sendKeys,
  typeText,
  press,
  scrollTo,
  clickRect,
  selectOption,
  submit,
  snapshot,
  find,
  BBox,
  Element,
  uncheck,
  uploadFile,
} from '../src';
import {
  createTestBrowser,
  getPageOrThrow,
  patchExampleDotCom,
  patchSearchEnginePages,
} from './test-utils';
import * as fs from 'fs';
import * as os from 'os';
import * as path from 'path';

describe('Actions', () => {
  describe('click', () => {
    it('should click an element by ID', async () => {
      const browser = await createTestBrowser();

      try {
        const page = getPageOrThrow(browser);
        await page.goto('https://example.com', { waitUntil: 'domcontentloaded', timeout: 20000 });

        const snap = await snapshot(browser, { screenshot: false, limit: 30 });

        const link = find(snap, 'role=link');

        if (link) {
          const result = await click(browser, link.id);
          expect(result.success).toBe(true);
          expect(result.duration_ms).toBeGreaterThan(0);
          expect(['navigated', 'dom_updated']).toContain(result.outcome);
        }
      } finally {
        await browser.close();
      }
    }, 60000);

    it('should use hybrid approach (mouse.click at center)', async () => {
      const browser = await createTestBrowser();

      try {
        const page = getPageOrThrow(browser);
        await page.goto('https://example.com', { waitUntil: 'domcontentloaded', timeout: 20000 });

        const snap = await snapshot(browser, { screenshot: false, limit: 30 });
        const link = find(snap, 'role=link');

        if (link) {
          // Test hybrid approach (mouse.click at center)
          const result = await click(browser, link.id, true);
          expect(result.success).toBe(true);
          expect(result.duration_ms).toBeGreaterThan(0);
          // Navigation may happen, which is expected for links
          expect(['navigated', 'dom_updated']).toContain(result.outcome);
        }
      } finally {
        await browser.close();
      }
    }, 60000);

    it('should use JS-based approach (legacy)', async () => {
      const browser = await createTestBrowser();

      try {
        const page = getPageOrThrow(browser);
        await page.goto('https://example.com', { waitUntil: 'domcontentloaded', timeout: 20000 });

        const snap = await snapshot(browser, { screenshot: false, limit: 30 });
        const link = find(snap, 'role=link');

        if (link) {
          // Test JS-based click (legacy approach)
          const result = await click(browser, link.id, false);
          expect(result.success).toBe(true);
          expect(result.duration_ms).toBeGreaterThan(0);
          // Navigation may happen, which is expected for links
          expect(['navigated', 'dom_updated']).toContain(result.outcome);
        }
      } finally {
        await browser.close();
      }
    }, 60000);
  });

  describe('typeText', () => {
    it('should type text into an element', async () => {
      const browser = await createTestBrowser();

      try {
        const page = getPageOrThrow(browser);
        await page.goto('https://example.com');
        await page.waitForLoadState('networkidle', { timeout: 10000 });

        const snap = await snapshot(browser);
        const textbox = find(snap, 'role=textbox');

        if (textbox) {
          const result = await typeText(browser, textbox.id, 'hello');
          expect(result.success).toBe(true);
          expect(result.duration_ms).toBeGreaterThan(0);
        }
      } finally {
        await browser.close();
      }
    }, 60000);
  });

  describe('press', () => {
    it('should press a keyboard key', async () => {
      const browser = await createTestBrowser();

      try {
        const page = getPageOrThrow(browser);
        await page.goto('https://example.com');
        await page.waitForLoadState('networkidle', { timeout: 10000 });

        const result = await press(browser, 'Enter');
        expect(result.success).toBe(true);
        expect(result.duration_ms).toBeGreaterThan(0);
      } finally {
        await browser.close();
      }
    }, 60000);
  });

  describe('sendKeys', () => {
    it('should send key sequences', async () => {
      const browser = await createTestBrowser();
      try {
        const page = getPageOrThrow(browser);
        await page.goto('https://example.com');
        await page.waitForLoadState('networkidle', { timeout: 10000 });

        const result = await sendKeys(browser, 'CTRL+L');
        expect(result.success).toBe(true);
        expect(result.duration_ms).toBeGreaterThan(0);
      } finally {
        await browser.close();
      }
    }, 60000);

    it('should throw on empty sequence', async () => {
      const browser = await createTestBrowser();
      try {
        const page = getPageOrThrow(browser);
        await page.goto('https://example.com');
        await page.waitForLoadState('networkidle', { timeout: 10000 });

        await expect(sendKeys(browser, '')).rejects.toThrow('empty');
      } finally {
        await browser.close();
      }
    }, 60000);
  });

  describe('search', () => {
    it('should build search URLs', async () => {
      const browser = await createTestBrowser();
      try {
        const page = getPageOrThrow(browser);
        await page.goto('https://example.com');
        await page.waitForLoadState('networkidle', { timeout: 10000 });

        await patchSearchEnginePages(page);

        const result = await search(browser, 'sentience sdk', 'duckduckgo');
        expect(result.success).toBe(true);
        expect(result.duration_ms).toBeGreaterThan(0);
        expect(page.url()).toBe('https://duckduckgo.com/?q=sentience+sdk');

        expect((await search(browser, 'sentience sdk', 'google')).success).toBe(true);
        expect(page.url()).toBe('https://www.google.com/search?q=sentience+sdk');

        expect((await search(browser, 'sentience sdk', 'bing')).success).toBe(true);
        expect(page.url()).toBe('https://www.bing.com/search?q=sentience+sdk');

        expect((await search(browser, 'sentience sdk', 'google.com')).success).toBe(true);
        expect(page.url()).toBe('https://www.google.com/search?q=sentience+sdk');
      } finally {
        await browser.close();
      }
    }, 90000);

    it('should reject empty query', async () => {
      const browser = await createTestBrowser();
      try {
        const page = getPageOrThrow(browser);
        await page.goto('https://example.com');
        await page.waitForLoadState('networkidle', { timeout: 10000 });

        await expect(search(browser, '')).rejects.toThrow('empty');
      } finally {
        await browser.close();
      }
    }, 60000);

    it('should reject disallowed domains', async () => {
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
        ['example.com']
      );
      try {
        await browser.start();
        const page = getPageOrThrow(browser);
        await patchExampleDotCom(page);
        await page.goto('https://example.com');

        await expect(search(browser, 'sentience sdk', 'duckduckgo')).rejects.toThrow(
          'domain not allowed'
        );
      } finally {
        await browser.close();
      }
    }, 60000);
  });

  describe('scrollTo', () => {
    it('should scroll an element into view', async () => {
      const browser = await createTestBrowser();

      try {
        const page = getPageOrThrow(browser);
        await page.goto('https://example.com');
        await page.waitForLoadState('networkidle', { timeout: 10000 });

        const snap = await snapshot(browser);
        // Find an element to scroll to
        const elements = snap.elements.filter((el: Element) => el.role === 'link');

        if (elements.length > 0) {
          // Get the last element which might be out of viewport
          const element = elements.length > 1 ? elements[elements.length - 1] : elements[0];
          const result = await scrollTo(browser, element.id);
          expect(result.success).toBe(true);
          expect(result.duration_ms).toBeGreaterThan(0);
          expect(['navigated', 'dom_updated']).toContain(result.outcome);
        }
      } finally {
        await browser.close();
      }
    }, 60000);

    it('should scroll with instant behavior', async () => {
      const browser = await createTestBrowser();

      try {
        const page = getPageOrThrow(browser);
        await page.goto('https://example.com');
        await page.waitForLoadState('networkidle', { timeout: 10000 });

        const snap = await snapshot(browser);
        const elements = snap.elements.filter((el: Element) => el.role === 'link');

        if (elements.length > 0) {
          const element = elements[0];
          const result = await scrollTo(browser, element.id, 'instant', 'start');
          expect(result.success).toBe(true);
          expect(result.duration_ms).toBeGreaterThan(0);
        }
      } finally {
        await browser.close();
      }
    }, 90000); // 90 seconds - Windows CI can be slow

    it('should take snapshot after scroll when requested', async () => {
      const browser = await createTestBrowser();

      try {
        const page = getPageOrThrow(browser);
        await page.goto('https://example.com');
        await page.waitForLoadState('networkidle', { timeout: 10000 });

        const snap = await snapshot(browser);
        const elements = snap.elements.filter((el: Element) => el.role === 'link');

        if (elements.length > 0) {
          const element = elements[0];
          const result = await scrollTo(browser, element.id, 'smooth', 'center', true);
          expect(result.success).toBe(true);
          expect(result.snapshot_after).toBeDefined();
          expect(result.snapshot_after?.status).toBe('success');
        }
      } finally {
        await browser.close();
      }
    }, 60000);

    it('should fail for invalid element ID', async () => {
      const browser = await createTestBrowser();

      try {
        const page = getPageOrThrow(browser);
        await page.goto('https://example.com');
        await page.waitForLoadState('networkidle', { timeout: 10000 });

        // Try to scroll to non-existent element
        const result = await scrollTo(browser, 99999);
        expect(result.success).toBe(false);
        expect(result.error).toBeDefined();
        expect(result.error?.code).toBe('scroll_failed');
      } finally {
        await browser.close();
      }
    }, 60000);
  });

  describe('typeText with delay', () => {
    it('should type text with human-like delay', async () => {
      const browser = await createTestBrowser();

      try {
        const page = getPageOrThrow(browser);
        await page.goto('https://example.com');
        await page.waitForLoadState('networkidle', { timeout: 10000 });

        const snap = await snapshot(browser);
        const textbox = find(snap, 'role=textbox');

        if (textbox) {
          // Test with 10ms delay between keystrokes
          const result = await typeText(browser, textbox.id, 'hello', false, 10);
          expect(result.success).toBe(true);
          // Duration should be longer due to delays (at least 5 chars * 10ms = 50ms)
          expect(result.duration_ms).toBeGreaterThanOrEqual(50);
        }
      } finally {
        await browser.close();
      }
    }, 60000);
  });

  describe('clickRect', () => {
    it('should click at rectangle center using rect dict', async () => {
      const browser = await createTestBrowser();

      try {
        const page = getPageOrThrow(browser);
        await page.goto('https://example.com');
        await page.waitForLoadState('networkidle', { timeout: 10000 });

        // Click at a specific rectangle (top-left area)
        const result = await clickRect(browser, { x: 100, y: 100, w: 50, h: 30 });
        expect(result.success).toBe(true);
        expect(result.duration_ms).toBeGreaterThan(0);
        expect(['navigated', 'dom_updated']).toContain(result.outcome);
      } finally {
        await browser.close();
      }
    }, 60000);

    it('should click using BBox object', async () => {
      const browser = await createTestBrowser();

      try {
        const page = getPageOrThrow(browser);
        await page.goto('https://example.com');
        await page.waitForLoadState('networkidle', { timeout: 10000 });

        // Get an element and click its bbox
        const snap = await snapshot(browser);
        const link = find(snap, 'role=link');

        if (link) {
          const bbox: BBox = {
            x: link.bbox.x,
            y: link.bbox.y,
            width: link.bbox.width,
            height: link.bbox.height,
          };
          const result = await clickRect(browser, bbox);
          expect(result.success).toBe(true);
          expect(result.duration_ms).toBeGreaterThan(0);
        }
      } finally {
        await browser.close();
      }
    }, 60000);

    it('should click without highlight when disabled', async () => {
      const browser = await createTestBrowser();

      try {
        const page = getPageOrThrow(browser);
        await page.goto('https://example.com');
        await page.waitForLoadState('networkidle', { timeout: 10000 });

        const result = await clickRect(browser, { x: 100, y: 100, w: 50, h: 30 }, false);
        expect(result.success).toBe(true);
        expect(result.duration_ms).toBeGreaterThan(0);
      } finally {
        await browser.close();
      }
    }, 60000);

    it('should handle invalid rectangle dimensions', async () => {
      const browser = await createTestBrowser();

      try {
        const page = getPageOrThrow(browser);
        await page.goto('https://example.com');
        await page.waitForLoadState('networkidle', { timeout: 10000 });

        // Invalid: zero width
        const result1 = await clickRect(browser, { x: 100, y: 100, w: 0, h: 30 });
        expect(result1.success).toBe(false);
        expect(result1.error).toBeDefined();
        expect(result1.error?.code).toBe('invalid_rect');

        // Invalid: negative height
        const result2 = await clickRect(browser, { x: 100, y: 100, w: 50, h: -10 });
        expect(result2.success).toBe(false);
        expect(result2.error).toBeDefined();
        expect(result2.error?.code).toBe('invalid_rect');
      } finally {
        await browser.close();
      }
    }, 60000);

    it('should take snapshot after click when requested', async () => {
      const browser = await createTestBrowser();

      try {
        const page = getPageOrThrow(browser);
        await page.goto('https://example.com');
        await page.waitForLoadState('networkidle', { timeout: 10000 });

        // Check if extension is available by trying to take a snapshot
        const snap = await snapshot(browser);

        const result = await clickRect(browser, { x: 100, y: 100, w: 50, h: 30 }, true, 2.0, true);
        expect(result.success).toBe(true);
        expect(result.snapshot_after).toBeDefined();
        expect(result.snapshot_after?.status).toBe('success');
        expect(result.snapshot_after?.elements.length).toBeGreaterThan(0);
      } finally {
        await browser.close();
      }
    }, 60000);

    it('should accept width/height keys instead of w/h', async () => {
      const browser = await createTestBrowser();

      try {
        const page = getPageOrThrow(browser);
        await page.goto('https://example.com');
        await page.waitForLoadState('networkidle', { timeout: 10000 });

        const result = await clickRect(browser, { x: 100, y: 100, width: 50, height: 30 });
        expect(result.success).toBe(true);
        expect(result.duration_ms).toBeGreaterThan(0);
      } finally {
        await browser.close();
      }
    }, 60000);
  });

  describe('CRUD helpers', () => {
    it('should clear/check/uncheck/select/upload/submit (best-effort)', async () => {
      const browser = await createTestBrowser();
      try {
        const page = getPageOrThrow(browser);
        await page.goto('https://example.com', { waitUntil: 'domcontentloaded', timeout: 20000 });
        await page.setContent(`
          <html><body>
            <input id="t" value="hello" />
            <input id="cb" type="checkbox" />
            <select id="sel">
              <option value="a">Alpha</option>
              <option value="b">Beta</option>
            </select>
            <form id="f">
              <input id="file" type="file" />
              <button id="btn" type="submit">Submit</button>
            </form>
            <script>
              window._submitted = false;
              document.getElementById('f').addEventListener('submit', (e) => {
                e.preventDefault();
                window._submitted = true;
              });
            </script>
          </body></html>
        `);

        await snapshot(browser, { screenshot: false, limit: 50 });

        const idOf = async (predSrc: string): Promise<number> => {
          const id = await page.evaluate((src: string) => {
            const reg = (window as any).sentience_registry || {};
            const pred = eval(src) as (el: any) => boolean; // test-only
            for (const [id, el] of Object.entries(reg)) {
              try {
                if (pred(el)) return Number(id);
              } catch {}
            }
            return null;
          }, predSrc);
          if (typeof id !== 'number') throw new Error('id not found');
          return id;
        };

        const tid = await idOf("(el) => el && el.id === 't'");
        const cbid = await idOf("(el) => el && el.id === 'cb'");
        const selid = await idOf("(el) => el && el.id === 'sel'");
        const fileid = await idOf("(el) => el && el.id === 'file'");
        const btnid = await idOf("(el) => el && el.id === 'btn'");

        expect((await clear(browser, tid)).success).toBe(true);
        expect(
          await page.evaluate(() => (document.getElementById('t') as HTMLInputElement).value)
        ).toBe('');

        expect((await check(browser, cbid)).success).toBe(true);
        expect(
          await page.evaluate(() => (document.getElementById('cb') as HTMLInputElement).checked)
        ).toBe(true);

        expect((await uncheck(browser, cbid)).success).toBe(true);
        expect(
          await page.evaluate(() => (document.getElementById('cb') as HTMLInputElement).checked)
        ).toBe(false);

        expect((await selectOption(browser, selid, 'b')).success).toBe(true);
        expect(
          await page.evaluate(() => (document.getElementById('sel') as HTMLSelectElement).value)
        ).toBe('b');

        const tmp = path.join(os.tmpdir(), `sentience-upload-${Date.now()}.txt`);
        fs.writeFileSync(tmp, 'hi', 'utf8');
        expect((await uploadFile(browser, fileid, tmp)).success).toBe(true);
        expect(
          await page.evaluate(
            () => (document.getElementById('file') as HTMLInputElement).files?.[0]?.name
          )
        ).toBe(path.basename(tmp));

        expect((await submit(browser, btnid)).success).toBe(true);
        expect(await page.evaluate(() => (window as any)._submitted)).toBe(true);

        // back() best-effort: just ensure it returns
        const r = await back(browser);
        expect(r.duration_ms).toBeGreaterThanOrEqual(0);
      } finally {
        await browser.close();
      }
    }, 60000);
  });
});
