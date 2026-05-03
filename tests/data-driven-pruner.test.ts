import { pruneWithPolicy } from '../src/agents/planner-executor/data-driven-pruner';
import type {
  DataDrivenPruningPolicy,
  LearnedTargetFingerprint,
} from '../src/agents/planner-executor/profile-types';
import { PruningTaskCategory } from '../src/agents/planner-executor/pruning-types';
import type { Snapshot, SnapshotElement } from '../src/agents/planner-executor/plan-models';

describe('pruneWithPolicy', () => {
  const basePolicy: DataDrivenPruningPolicy = {
    allowedRoles: ['button', 'link', 'textbox', 'searchbox'],
    includeTextPatterns: ['search', 'find', 'go'],
    excludeTextPatterns: ['privacy', 'terms', 'cookie'],
    maxElements: 10,
    maxElementsRelaxed: 20,
    maxElementsLoose: 30,
  };

  function makeSnapshot(elements: Partial<SnapshotElement>[]): Snapshot {
    return {
      url: 'https://example.com',
      title: 'Test Page',
      elements: elements.map((e, i) => ({
        id: String(i),
        role: 'button',
        text: '',
        ...e,
      })) as SnapshotElement[],
    };
  }

  describe('maxNodes by relaxation level', () => {
    it('should use maxElements at relaxation level 0', () => {
      const snapshot = makeSnapshot([]);
      const result = pruneWithPolicy(snapshot, basePolicy, 'search', 0, PruningTaskCategory.SEARCH);
      expect(result.maxNodes).toBe(10);
    });

    it('should use maxElementsRelaxed at relaxation level 1', () => {
      const snapshot = makeSnapshot([]);
      const result = pruneWithPolicy(snapshot, basePolicy, 'search', 1, PruningTaskCategory.SEARCH);
      expect(result.maxNodes).toBe(20);
    });

    it('should use maxElementsLoose at relaxation level 2+', () => {
      const snapshot = makeSnapshot([]);
      const result = pruneWithPolicy(snapshot, basePolicy, 'search', 2, PruningTaskCategory.SEARCH);
      expect(result.maxNodes).toBe(30);
    });

    it('should fall back when maxElementsLoose not set', () => {
      const noLoose = { ...basePolicy, maxElementsLoose: undefined };
      const snapshot = makeSnapshot([]);
      const result = pruneWithPolicy(snapshot, noLoose, 'search', 3, PruningTaskCategory.SEARCH);
      expect(result.maxNodes).toBe(40); // min(maxElementsRelaxed * 2, 100)
    });
  });

  describe('element filtering', () => {
    it('should filter to allowed roles', () => {
      const snapshot = makeSnapshot([
        { role: 'button', text: 'Search' },
        { role: 'img', text: 'Search icon' },
        { role: 'link', text: 'Find more' },
      ]);
      const result = pruneWithPolicy(snapshot, basePolicy, 'search', 0, PruningTaskCategory.SEARCH);
      const roles = result.elements.map(e => e.role);
      expect(roles).not.toContain('img');
      expect(roles).toContain('button');
      expect(roles).toContain('link');
    });

    it('should exclude elements matching exclude patterns', () => {
      const snapshot = makeSnapshot([
        { role: 'button', text: 'Search' },
        { role: 'link', text: 'Privacy Policy' },
        { role: 'link', text: 'Cookie Settings' },
      ]);
      const result = pruneWithPolicy(snapshot, basePolicy, 'search', 0, PruningTaskCategory.SEARCH);
      const texts = result.elements.map(e => e.text);
      expect(texts).not.toContain('Privacy Policy');
      expect(texts).not.toContain('Cookie Settings');
    });

    it('should require include patterns match when specified', () => {
      const snapshot = makeSnapshot([
        { role: 'button', text: 'Search' },
        { role: 'button', text: 'Random' },
      ]);
      const result = pruneWithPolicy(snapshot, basePolicy, 'search', 0, PruningTaskCategory.SEARCH);
      const texts = result.elements.map(e => e.text);
      expect(texts).toContain('Search');
      expect(texts).not.toContain('Random');
    });
  });

  describe('fingerprint boosting', () => {
    it('should boost elements matching learned fingerprints', () => {
      const policyNoInclude: DataDrivenPruningPolicy = {
        allowedRoles: ['button'],
        maxElements: 2,
        maxElementsRelaxed: 4,
      };

      const fingerprints: LearnedTargetFingerprint[] = [
        {
          domain: 'example.com',
          taskHash: 'abc',
          intent: 'search',
          role: 'button',
          textTokens: ['go'],
          successCount: 3,
          failureCount: 0,
          confidence: 0.9,
          learnedAt: Date.now(),
        },
      ];

      const snapshot = makeSnapshot([
        { role: 'button', text: 'Go', clickable: true },
        { role: 'button', text: 'Other', clickable: true },
      ]);

      const result = pruneWithPolicy(
        snapshot,
        policyNoInclude,
        'search',
        0,
        PruningTaskCategory.SEARCH,
        fingerprints
      );
      // "Go" button should be first due to fingerprint boost
      expect(result.elements[0].text).toBe('Go');
    });
  });
});
