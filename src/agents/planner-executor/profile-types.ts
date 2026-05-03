/**
 * Profile types for extensible task categories and task learning.
 *
 * @see docs/plans/browser-agent/2026-05-02-extensible-categories-and-task-learning.md
 */

import type { TaskCategory } from './task-category';
import type { HeuristicHintInput } from './heuristic-hint';

// ---------------------------------------------------------------------------
// Data-Driven Pruning Policy
// ---------------------------------------------------------------------------

export interface DataDrivenPruningPolicy {
  /** Element roles to include (lowercased, e.g. "button", "link") */
  allowedRoles: string[];
  /** Text patterns to include (case-insensitive substring) */
  includeTextPatterns?: string[];
  /** Text patterns to exclude (case-insensitive substring) */
  excludeTextPatterns?: string[];
  /** Max elements at relaxation level 0 */
  maxElements: number;
  /** Max elements at relaxation level 1 */
  maxElementsRelaxed: number;
  /** Max elements at relaxation level 2+ */
  maxElementsLoose?: number;
}

// ---------------------------------------------------------------------------
// Browser Agent Profile
// ---------------------------------------------------------------------------

export interface BrowserAgentProfile {
  /** Unique profile identifier */
  id: string;

  /** Display label for UI */
  label: string;

  /** Schema version for forward compatibility */
  version: number;

  /** When and how this profile should be activated */
  match: {
    /** Keywords found in task/goal text (case-insensitive substring) */
    taskKeywords?: string[];
    /** URL hostname patterns (glob-style, e.g., "*.booking.com") */
    domainPatterns?: string[];
  };

  /** Optional hint to a built-in TaskCategory for SDK compatibility */
  taskCategoryHint?: TaskCategory;

  /** Custom pruning policy (used when taskCategoryHint is absent or insufficient) */
  pruningPolicy?: DataDrivenPruningPolicy;

  /** Content-script scoring profile overrides */
  scoringProfile?: Record<string, unknown>;

  /** Domain-specific heuristic hints */
  heuristicHints?: HeuristicHintInput[];

  /** Where this profile came from */
  source: 'built_in' | 'user' | 'learned' | 'imported';

  /** Priority when multiple profiles match (higher = checked first) */
  priority?: number;
}

// ---------------------------------------------------------------------------
// Resolved Agent Profile (runtime composition)
// ---------------------------------------------------------------------------

export interface ResolvedAgentProfile {
  /** Optional built-in category hint */
  categoryHint?: TaskCategory;

  /** Data-driven pruning policy (from profile or learned) */
  pruningPolicy?: DataDrivenPruningPolicy;

  /** Content-script scoring profile overrides */
  scoringProfile?: Record<string, unknown>;

  /** Heuristic hints from profiles and learned fingerprints */
  heuristicHints: HeuristicHintInput[];

  /** Learned target fingerprints validated against current snapshot */
  learnedFingerprints: LearnedTargetFingerprint[];
}

// ---------------------------------------------------------------------------
// Learned Target Fingerprints (L1 Learning)
// ---------------------------------------------------------------------------

export interface LearnedTargetFingerprint {
  /** Domain (e.g., "amazon.com") */
  domain: string;

  /** Hash of the task goal for task-scoped matching */
  taskHash: string;

  /** The intent that was being resolved */
  intent: string;

  /** Element role */
  role?: string;

  /** Normalized visible text tokens (bounded, redacted) */
  textTokens?: string[];

  /** Normalized aria/name tokens */
  ariaTokens?: string[];

  /** Sanitized href path pattern (e.g., "/dp/", "/s?k=") */
  hrefPathPattern?: string;

  /** Stable attribute patterns */
  attributePatterns?: Record<string, string>;

  /** How many times this fingerprint led to a successful action */
  successCount: number;

  /** How many times this fingerprint was tried and failed */
  failureCount: number;

  /** Computed confidence (0-1), derived from success/failure counts and recency */
  confidence: number;

  /** When this fingerprint was first learned */
  learnedAt: number;

  /** When this fingerprint was last successfully used */
  lastUsedAt?: number;
}

// ---------------------------------------------------------------------------
// Domain Profile (L3 Learning)
// ---------------------------------------------------------------------------

export interface DomainProfile {
  /** Domain */
  domain: string;

  /** Preferred built-in TaskCategory hint */
  preferredCategoryHint?: TaskCategory;

  /** Preferred custom pruning policy (learned from successful runs) */
  preferredPruningPolicy?: DataDrivenPruningPolicy;

  /** Snapshot limit that worked */
  preferredSnapshotLimit: number;

  /** Average relaxation level needed */
  avgRelaxationLevel: number;

  /** Common intents seen on this domain */
  commonIntents: string[];

  /** Number of completed runs on this domain */
  runCount: number;

  /** Success rate (0-1) */
  successRate: number;

  /** Last updated */
  updatedAt: number;
}

// ---------------------------------------------------------------------------
// Empty / default helpers
// ---------------------------------------------------------------------------

export const EMPTY_RESOLVED_PROFILE: ResolvedAgentProfile = {
  heuristicHints: [],
  learnedFingerprints: [],
};
