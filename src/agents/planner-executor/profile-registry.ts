/**
 * ProfileRegistry — registers, resolves, and manages BrowserAgentProfiles.
 *
 * Resolution order:
 * 1. Sort profiles by priority descending (default 0)
 * 2. For each profile: check if any taskKeywords appear in taskText (case-insensitive)
 *    AND any domainPatterns match the domain (glob-style)
 * 3. First matching profile wins → extract hints, pruning, scoring
 * 4. If no profile matches → return EMPTY_RESOLVED_PROFILE
 *
 * @see docs/plans/browser-agent/2026-05-02-extensible-categories-and-task-learning.md
 */

import type {
  BrowserAgentProfile,
  ResolvedAgentProfile,
  LearnedTargetFingerprint,
} from './profile-types';
import { EMPTY_RESOLVED_PROFILE } from './profile-types';
import { BrowserAgentProfileSchema, BrowserAgentProfileArraySchema } from './profile-schema';
import type { TaskCategory } from './task-category';

export class ProfileRegistry {
  private profiles: BrowserAgentProfile[] = [];

  /** Register a profile */
  register(profile: BrowserAgentProfile): void {
    const existing = this.profiles.findIndex(p => p.id === profile.id);
    if (existing >= 0) {
      this.profiles[existing] = profile;
    } else {
      this.profiles.push(profile);
    }
  }

  /** Load one or more profiles from validated JSON */
  loadFromJSON(json: unknown): { loaded: number; errors: string[] } {
    const errors: string[] = [];
    let loaded = 0;

    // Try as array first, then as single object
    let items: unknown[];
    const arrayResult = BrowserAgentProfileArraySchema.safeParse(json);
    if (arrayResult.success) {
      items = arrayResult.data;
    } else {
      const singleResult = BrowserAgentProfileSchema.safeParse(json);
      if (singleResult.success) {
        items = [singleResult.data];
      } else {
        return {
          loaded: 0,
          errors: ['Invalid profile JSON: must be a profile object or array of profiles'],
        };
      }
    }

    for (let i = 0; i < items.length; i++) {
      const result = BrowserAgentProfileSchema.safeParse(items[i]);
      if (result.success) {
        // Zod enum produces string literals; cast to BrowserAgentProfile
        // whose taskCategoryHint uses the TaskCategory enum
        this.register(result.data as unknown as BrowserAgentProfile);
        loaded++;
      } else {
        errors.push(`Profile at index ${i}: ${result.error.message}`);
      }
    }

    return { loaded, errors };
  }

  /** Resolve the best matching profile for a task + domain */
  resolve(
    taskText: string,
    domain: string,
    fingerprints?: LearnedTargetFingerprint[]
  ): ResolvedAgentProfile {
    const sorted = [...this.profiles].sort((a, b) => (b.priority ?? 0) - (a.priority ?? 0));
    const normalizedTask = taskText.toLowerCase();

    for (const profile of sorted) {
      if (this.matchesProfile(profile, normalizedTask, domain)) {
        return {
          categoryHint: profile.taskCategoryHint,
          pruningPolicy: profile.pruningPolicy,
          scoringProfile: profile.scoringProfile,
          heuristicHints: profile.heuristicHints ?? [],
          learnedFingerprints: fingerprints ?? [],
        };
      }
    }

    return {
      ...EMPTY_RESOLVED_PROFILE,
      learnedFingerprints: fingerprints ?? [],
    };
  }

  /** List all registered profiles */
  list(): BrowserAgentProfile[] {
    return [...this.profiles];
  }

  /** Remove a profile by id */
  unregister(id: string): boolean {
    const index = this.profiles.findIndex(p => p.id === id);
    if (index < 0) return false;
    this.profiles.splice(index, 1);
    return true;
  }

  /** Clear all profiles */
  clear(): void {
    this.profiles = [];
  }

  private matchesProfile(
    profile: BrowserAgentProfile,
    normalizedTask: string,
    domain: string
  ): boolean {
    const { taskKeywords, domainPatterns } = profile.match;

    const keywordsMatch =
      !taskKeywords ||
      taskKeywords.length === 0 ||
      taskKeywords.some(kw => normalizedTask.includes(kw.toLowerCase()));

    const domainMatch =
      !domainPatterns ||
      domainPatterns.length === 0 ||
      domainPatterns.some(pattern => globMatch(pattern, domain));

    return keywordsMatch && domainMatch;
  }
}

/**
 * Simple glob-style matching for domain patterns.
 * Supports "*" as a wildcard (e.g., "*.booking.com" matches "www.booking.com").
 */
function globMatch(pattern: string, domain: string): boolean {
  const normalizedPattern = pattern.toLowerCase();
  const normalizedDomain = domain.toLowerCase();

  if (!normalizedPattern.includes('*')) {
    return (
      normalizedDomain === normalizedPattern || normalizedDomain.endsWith('.' + normalizedPattern)
    );
  }

  // Convert glob to regex: escape dots, replace * with .*
  const regexStr = normalizedPattern.replace(/[.+^${}()|[\]\\]/g, '\\$&').replace(/\*/g, '.*');
  try {
    return new RegExp(`^${regexStr}$`).test(normalizedDomain);
  } catch {
    return false;
  }
}
