import { ProfileRegistry } from '../src/agents/planner-executor/profile-registry';
import type { BrowserAgentProfile } from '../src/agents/planner-executor/profile-types';
import { TaskCategory } from '../src/agents/planner-executor/task-category';

describe('ProfileRegistry', () => {
  let registry: ProfileRegistry;

  beforeEach(() => {
    registry = new ProfileRegistry();
  });

  describe('register()', () => {
    it('should register a valid profile', () => {
      const profile = makeProfile('test-1', 'Test Profile', {
        taskKeywords: ['search'],
      });
      registry.register(profile);
      expect(registry.list()).toHaveLength(1);
    });

    it('should list registered profiles', () => {
      registry.register(makeProfile('a', 'Profile A'));
      registry.register(makeProfile('b', 'Profile B'));
      expect(registry.list()).toHaveLength(2);
    });
  });

  describe('unregister()', () => {
    it('should remove a profile by id', () => {
      registry.register(makeProfile('test-1', 'Test'));
      expect(registry.unregister('test-1')).toBe(true);
      expect(registry.list()).toHaveLength(0);
    });

    it('should return false for unknown id', () => {
      expect(registry.unregister('nonexistent')).toBe(false);
    });
  });

  describe('resolve()', () => {
    it('should match profile by task keyword', () => {
      const profile = makeProfile('shopping', 'Shopping', {
        taskKeywords: ['cart', 'buy', 'shop'],
      });
      registry.register(profile);

      const result = registry.resolve('Search for laptops and add to cart', 'example.com');
      expect(result.categoryHint).toBe('transaction');
    });

    it('should match profile by domain pattern', () => {
      const profile = makeProfile('travel', 'Travel', {
        domainPatterns: ['*.booking.com'],
      });
      profile.taskCategoryHint = TaskCategory.SEARCH;
      registry.register(profile);

      const result = registry.resolve('Book a hotel', 'www.booking.com');
      expect(result.categoryHint).toBe('search');
    });

    it('should prefer higher priority profiles', () => {
      const lowPriority = makeProfile('low', 'Low Priority', { taskKeywords: ['search'] });
      lowPriority.priority = 1;
      lowPriority.pruningPolicy = {
        allowedRoles: ['button'],
        maxElements: 10,
        maxElementsRelaxed: 20,
      };

      const highPriority = makeProfile('high', 'High Priority', { taskKeywords: ['search'] });
      highPriority.priority = 100;
      highPriority.pruningPolicy = {
        allowedRoles: ['link'],
        maxElements: 5,
        maxElementsRelaxed: 15,
      };

      registry.register(lowPriority);
      registry.register(highPriority);

      const result = registry.resolve('search for something', 'example.com');
      expect(result.pruningPolicy?.allowedRoles).toContain('link');
    });

    it('should return empty profile when nothing matches', () => {
      registry.register(makeProfile('travel', 'Travel', { taskKeywords: ['flight'] }));

      const result = registry.resolve('search for laptops', 'example.com');
      expect(result.categoryHint).toBeUndefined();
      expect(result.heuristicHints).toEqual([]);
    });

    it('should pass through learned fingerprints', () => {
      registry.register(makeProfile('base', 'Base', { taskKeywords: ['search'] }));

      const fingerprints = [makeFingerprint('example.com', 'abc123', 'search', 0.9)];

      const result = registry.resolve('search for something', 'example.com', fingerprints);
      expect(result.learnedFingerprints).toHaveLength(1);
    });
  });

  describe('loadFromJSON()', () => {
    it('should load valid profile from JSON array', () => {
      const json = [
        {
          id: 'imported-1',
          label: 'Imported Profile',
          version: 1,
          match: { taskKeywords: ['book'] },
          source: 'imported',
        },
      ];

      const result = registry.loadFromJSON(json);
      expect(result.loaded).toBe(1);
      expect(result.errors).toHaveLength(0);
    });

    it('should report errors for invalid input', () => {
      const result = registry.loadFromJSON([
        {
          /* missing required fields */
        },
      ]);
      expect(result.loaded).toBe(0);
      expect(result.errors.length).toBeGreaterThan(0);
    });

    it('should handle non-array input', () => {
      const result = registry.loadFromJSON('not an array');
      expect(result.loaded).toBe(0);
    });
  });
});

// ---------------------------------------------------------------------------
// Helpers
// ---------------------------------------------------------------------------

function makeProfile(
  id: string,
  label: string,
  match?: { taskKeywords?: string[]; domainPatterns?: string[] }
): BrowserAgentProfile {
  return {
    id,
    label,
    version: 1,
    match: match || {},
    taskCategoryHint: TaskCategory.TRANSACTION,
    source: 'user',
    priority: 10,
  };
}

function makeFingerprint(domain: string, taskHash: string, intent: string, confidence: number) {
  return {
    domain,
    taskHash,
    intent,
    role: 'button' as const,
    textTokens: ['click', 'me'],
    successCount: 1,
    failureCount: 0,
    confidence,
    learnedAt: Date.now(),
  };
}
