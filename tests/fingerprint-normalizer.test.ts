import {
  computeTaskHash,
  extractDomain,
  createFingerprint,
  mergeFingerprint,
  recordFingerprintFailure,
} from '../src/agents/planner-executor/fingerprint-normalizer';
import type { LearnedTargetFingerprint } from '../src/agents/planner-executor/profile-types';

describe('fingerprint-normalizer', () => {
  describe('computeTaskHash', () => {
    it('should produce a stable hash for the same input', () => {
      const h1 = computeTaskHash('Search for headphones on Amazon');
      const h2 = computeTaskHash('Search for headphones on Amazon');
      expect(h1).toBe(h2);
    });

    it('should normalize whitespace and case', () => {
      const h1 = computeTaskHash('  Search   for laptops  ');
      const h2 = computeTaskHash('search for laptops');
      expect(h1).toBe(h2);
    });

    it('should produce different hashes for different tasks', () => {
      const h1 = computeTaskHash('Search for laptops');
      const h2 = computeTaskHash('Book a flight to Paris');
      expect(h1).not.toBe(h2);
    });

    it('should return a hash prefixed with th_', () => {
      const hash = computeTaskHash('test task');
      expect(hash).toMatch(/^th_[a-z0-9]+$/);
    });
  });

  describe('extractDomain', () => {
    it('should extract hostname from a URL', () => {
      expect(extractDomain('https://www.example.com/path')).toBe('www.example.com');
    });

    it('should return empty string for invalid URL', () => {
      expect(extractDomain('not-a-url')).toBe('');
    });
  });

  describe('createFingerprint', () => {
    it('should create a fingerprint with normalized text tokens', () => {
      const element = {
        id: 1,
        role: 'button',
        text: 'Click here to search',
        name: '',
        href: '/search?q=test',
      } as any;
      const fp = createFingerprint(element, 'search', 'th_abc', 'example.com');

      expect(fp.domain).toBe('example.com');
      expect(fp.taskHash).toBe('th_abc');
      expect(fp.intent).toBe('search');
      expect(fp.role).toBe('button');
      expect(fp.textTokens).toEqual(['Click', 'here', 'to', 'search']);
      expect(fp.successCount).toBe(1);
      expect(fp.confidence).toBe(0.5);
    });

    it('should sanitize href to path only', () => {
      const element = {
        id: 1,
        role: 'link',
        text: 'Go',
        name: '',
        href: 'https://example.com/search?q=shoes&ref=nav',
      } as any;
      const fp = createFingerprint(element, 'navigate', 'th_abc', 'example.com');
      expect(fp.hrefPathPattern).toBe('/search');
    });

    it('should handle elements with no text', () => {
      const element = { id: 1, role: 'button', text: '', name: '', href: '' } as any;
      const fp = createFingerprint(element, 'click', 'th_abc', 'example.com');
      expect(fp.textTokens).toEqual([]);
    });
  });

  describe('mergeFingerprint', () => {
    it('should add new fingerprint to empty list', () => {
      const fp = makeFingerprint('example.com', 'th_1', 'search', 'button');
      const result = mergeFingerprint([], fp);
      expect(result).toHaveLength(1);
    });

    it('should update existing fingerprint on match', () => {
      const existing = [makeFingerprint('example.com', 'th_1', 'search', 'button')];
      existing[0].successCount = 1;
      existing[0].confidence = 0.5;

      const newFp = makeFingerprint('example.com', 'th_1', 'search', 'button');
      const result = mergeFingerprint(existing, newFp);

      expect(result).toHaveLength(1);
      expect(result[0].successCount).toBe(2);
      expect(result[0].confidence).toBeGreaterThan(0.5);
    });

    it('should add as new entry when no match', () => {
      const existing = [makeFingerprint('example.com', 'th_1', 'search', 'button')];
      const newFp = makeFingerprint('other.com', 'th_2', 'search', 'button');
      const result = mergeFingerprint(existing, newFp);

      expect(result).toHaveLength(2);
    });
  });

  describe('recordFingerprintFailure', () => {
    it('should increment failure count for matching fingerprint', () => {
      const existing = [makeFingerprint('example.com', 'th_1', 'search', 'button')];
      existing[0].successCount = 3;
      existing[0].confidence = 1.0;

      const result = recordFingerprintFailure(existing, 'search', 'th_1', 'example.com');
      expect(result[0].failureCount).toBe(1);
      expect(result[0].confidence).toBeLessThan(1.0);
    });

    it('should not modify list when no match', () => {
      const existing = [makeFingerprint('example.com', 'th_1', 'search', 'button')];
      const result = recordFingerprintFailure(existing, 'other', 'th_2', 'other.com');
      expect(result[0].failureCount).toBe(0);
    });
  });
});

function makeFingerprint(
  domain: string,
  taskHash: string,
  intent: string,
  role: string
): LearnedTargetFingerprint {
  return {
    domain,
    taskHash,
    intent,
    role,
    textTokens: ['click'],
    successCount: 1,
    failureCount: 0,
    confidence: 0.5,
    learnedAt: Date.now(),
  };
}
