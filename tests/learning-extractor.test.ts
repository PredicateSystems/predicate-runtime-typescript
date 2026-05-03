import {
  isSensitiveUrl,
  extractFingerprintFromOutcome,
  applyFingerprintFailure,
  applyFingerprintSuccess,
  isFingerprintStale,
  isFingerprintExpired,
  fingerprintToHint,
  computeTaskHash,
} from '../src/agents/planner-executor/learning-extractor';
import { StepStatus } from '../src/agents/planner-executor/plan-models';
import type { LearnedTargetFingerprint } from '../src/agents/planner-executor/profile-types';

function makeOutcome(overrides: Record<string, unknown> = {}) {
  return {
    stepId: 1,
    status: StepStatus.SUCCESS,
    actionTaken: 'CLICK(5)',
    goal: 'search for headphones',
    ...overrides,
  } as any;
}

function makeElements() {
  return [
    { id: 1, role: 'textbox', text: 'Search', name: 'q', href: '' },
    { id: 5, role: 'button', text: 'Search', name: 'btnSearch', href: '/search' },
    { id: 10, role: 'link', text: 'Add to Cart', name: '', href: '/cart/add' },
  ] as any[];
}

function makeFingerprint(
  overrides: Partial<LearnedTargetFingerprint> = {}
): LearnedTargetFingerprint {
  return {
    domain: 'example.com',
    taskHash: 'th_abc123',
    intent: 'search',
    role: 'button',
    textTokens: ['search'],
    successCount: 1,
    failureCount: 0,
    confidence: 0.5,
    learnedAt: Date.now(),
    ...overrides,
  };
}

describe('learning-extractor', () => {
  describe('isSensitiveUrl', () => {
    it('should detect checkout URLs as sensitive', () => {
      expect(isSensitiveUrl('https://shop.com/checkout')).toBe(true);
    });

    it('should detect payment URLs as sensitive', () => {
      expect(isSensitiveUrl('https://shop.com/payment')).toBe(true);
    });

    it('should detect login URLs as sensitive', () => {
      expect(isSensitiveUrl('https://example.com/login')).toBe(true);
    });

    it('should detect auth URLs as sensitive', () => {
      expect(isSensitiveUrl('https://example.com/auth/callback')).toBe(true);
    });

    it('should allow normal shopping URLs', () => {
      expect(isSensitiveUrl('https://amazon.com/product/laptop')).toBe(false);
    });

    it('should allow search URLs', () => {
      expect(isSensitiveUrl('https://google.com/search?q=test')).toBe(false);
    });
  });

  describe('extractFingerprintFromOutcome', () => {
    const defaultOptions = {
      learningEnabled: true,
      taskGoal: 'Search for headphones',
      currentUrl: 'https://example.com/search',
      taskHash: 'th_abc123',
    };

    it('should skip when learning is disabled', () => {
      const result = extractFingerprintFromOutcome(makeOutcome(), makeElements(), {
        ...defaultOptions,
        learningEnabled: false,
      });
      expect(result.extracted).toBe(false);
      expect(result.skipReason).toBe('learning_disabled');
    });

    it('should skip when step is not successful', () => {
      const result = extractFingerprintFromOutcome(
        makeOutcome({ status: StepStatus.FAILED }),
        makeElements(),
        defaultOptions
      );
      expect(result.extracted).toBe(false);
      expect(result.skipReason).toBe('step_not_successful');
    });

    it('should skip on sensitive URLs', () => {
      const result = extractFingerprintFromOutcome(makeOutcome(), makeElements(), {
        ...defaultOptions,
        currentUrl: 'https://shop.com/checkout',
      });
      expect(result.extracted).toBe(false);
      expect(result.skipReason).toBe('sensitive_url');
    });

    it('should skip when no snapshot elements', () => {
      const result = extractFingerprintFromOutcome(makeOutcome(), undefined, defaultOptions);
      expect(result.extracted).toBe(false);
      expect(result.skipReason).toBe('no_snapshot_elements');
    });

    it('should skip when action has no element ID (e.g., DONE)', () => {
      const result = extractFingerprintFromOutcome(
        makeOutcome({ actionTaken: 'DONE()' }),
        makeElements(),
        defaultOptions
      );
      expect(result.extracted).toBe(false);
      expect(result.skipReason).toBe('no_element_action');
    });

    it('should skip when element is not found in snapshot', () => {
      const result = extractFingerprintFromOutcome(
        makeOutcome({ actionTaken: 'CLICK(999)' }),
        makeElements(),
        defaultOptions
      );
      expect(result.extracted).toBe(false);
      expect(result.skipReason).toBe('element_not_found');
    });

    it('should skip when element is a password field', () => {
      const elements = [{ id: 5, role: 'password', text: 'Enter password', name: 'pw', href: '' }];
      const result = extractFingerprintFromOutcome(makeOutcome(), elements, defaultOptions);
      expect(result.extracted).toBe(false);
      expect(result.skipReason).toBe('sensitive_element');
    });

    it('should extract fingerprint from successful CLICK action', () => {
      const result = extractFingerprintFromOutcome(makeOutcome(), makeElements(), defaultOptions);
      expect(result.extracted).toBe(true);
      expect(result.fingerprint).toBeDefined();
      expect(result.fingerprint!.domain).toBe('example.com');
      expect(result.fingerprint!.intent).toBe('search_for_headphones');
      expect(result.fingerprint!.role).toBe('button');
    });

    it('should extract fingerprint from TYPE action', () => {
      const result = extractFingerprintFromOutcome(
        makeOutcome({ actionTaken: 'TYPE(1, "laptop")' }),
        makeElements(),
        defaultOptions
      );
      expect(result.extracted).toBe(true);
      expect(result.fingerprint!.role).toBe('textbox');
    });
  });

  describe('applyFingerprintFailure', () => {
    it('should increment failure count and reduce confidence', () => {
      const fp = makeFingerprint({ successCount: 3, failureCount: 0, confidence: 1.0 });
      const result = applyFingerprintFailure(fp);
      expect(result.failureCount).toBe(1);
      expect(result.confidence).toBeLessThan(1.0);
    });

    it('should apply exponential decay with multiple failures', () => {
      let fp = makeFingerprint({ successCount: 1, failureCount: 0, confidence: 1.0 });
      fp = applyFingerprintFailure(fp);
      const c1 = fp.confidence;
      fp = applyFingerprintFailure(fp);
      const c2 = fp.confidence;
      fp = applyFingerprintFailure(fp);
      const c3 = fp.confidence;
      // Confidence should decrease with each failure
      expect(c2).toBeLessThan(c1);
      expect(c3).toBeLessThan(c2);
    });

    it('should not go below 0 confidence', () => {
      const fp = makeFingerprint({ successCount: 0, failureCount: 100, confidence: 0 });
      const result = applyFingerprintFailure(fp);
      expect(result.confidence).toBeGreaterThanOrEqual(0);
    });
  });

  describe('applyFingerprintSuccess', () => {
    it('should increment success count and update lastUsedAt', () => {
      const fp = makeFingerprint({ successCount: 1, failureCount: 0 });
      const before = Date.now();
      const result = applyFingerprintSuccess(fp);
      expect(result.successCount).toBe(2);
      expect(result.lastUsedAt).toBeGreaterThanOrEqual(before);
    });

    it('should increase confidence with more successes', () => {
      const fp = makeFingerprint({ successCount: 1, failureCount: 2, confidence: 0.3 });
      const result = applyFingerprintSuccess(fp);
      expect(result.confidence).toBeGreaterThan(fp.confidence);
    });

    it('should cap confidence at 1', () => {
      const fp = makeFingerprint({ successCount: 100, failureCount: 0, confidence: 1.0 });
      const result = applyFingerprintSuccess(fp);
      expect(result.confidence).toBeLessThanOrEqual(1);
    });
  });

  describe('isFingerprintStale', () => {
    it('should return false for healthy fingerprint', () => {
      const fp = makeFingerprint({ failureCount: 1, confidence: 0.5 });
      expect(isFingerprintStale(fp)).toBe(false);
    });

    it('should return true for high failures and low confidence', () => {
      const fp = makeFingerprint({ failureCount: 5, confidence: 0.1 });
      expect(isFingerprintStale(fp)).toBe(true);
    });

    it('should return false when confidence is still reasonable despite failures', () => {
      const fp = makeFingerprint({ failureCount: 3, confidence: 0.5 });
      expect(isFingerprintStale(fp)).toBe(false);
    });
  });

  describe('isFingerprintExpired', () => {
    it('should return false for recent fingerprint', () => {
      const fp = makeFingerprint({ learnedAt: Date.now() - 1000 });
      expect(isFingerprintExpired(fp, 60000)).toBe(false);
    });

    it('should return true for old fingerprint', () => {
      const fp = makeFingerprint({ learnedAt: Date.now() - 120000 });
      expect(isFingerprintExpired(fp, 60000)).toBe(true);
    });
  });

  describe('fingerprintToHint', () => {
    it('should convert a high-confidence fingerprint to a hint', () => {
      const fp = makeFingerprint({ confidence: 0.8, textTokens: ['search'], ariaTokens: ['find'] });
      const hint = fingerprintToHint(fp);
      expect(hint).not.toBeNull();
      expect(hint!.intent).toBe('search');
      expect(hint!.textPatterns).toContain('search');
      expect(hint!.textPatterns).toContain('find');
      expect(hint!.priority).toBeGreaterThan(0);
    });

    it('should return null for low-confidence fingerprint', () => {
      const fp = makeFingerprint({ confidence: 0.2 });
      const hint = fingerprintToHint(fp);
      expect(hint).toBeNull();
    });

    it('should return null for stale fingerprint', () => {
      const fp = makeFingerprint({ confidence: 0.5, failureCount: 5 });
      // Make it stale: confidence < 0.3 and failureCount >= 3
      fp.confidence = 0.2;
      fp.failureCount = 4;
      const hint = fingerprintToHint(fp);
      expect(hint).toBeNull();
    });

    it('should respect custom minConfidence', () => {
      const fp = makeFingerprint({ confidence: 0.45 });
      expect(fingerprintToHint(fp, 0.5)).toBeNull();
      expect(fingerprintToHint(fp, 0.4)).not.toBeNull();
    });
  });

  describe('computeTaskHash', () => {
    it('should produce stable hashes', async () => {
      const h1 = await computeTaskHash('search for laptops');
      const h2 = await computeTaskHash('search for laptops');
      expect(h1).toBe(h2);
    });

    it('should normalize case and whitespace', async () => {
      const h1 = await computeTaskHash('  Search   For Laptops  ');
      const h2 = await computeTaskHash('search for laptops');
      expect(h1).toBe(h2);
    });

    it('should produce different hashes for different tasks', async () => {
      const h1 = await computeTaskHash('search for laptops');
      const h2 = await computeTaskHash('book a flight');
      expect(h1).not.toBe(h2);
    });
  });
});
