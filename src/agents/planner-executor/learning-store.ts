/**
 * LearningStore — interface for persisting and retrieving learned data.
 *
 * The SDK defines the interface; the extension provides the implementation
 * backed by chrome.storage.local. This keeps the SDK browser-agnostic.
 *
 * @see docs/plans/browser-agent/2026-05-02-extensible-categories-and-task-learning.md
 */

import type { LearnedTargetFingerprint, DomainProfile } from './profile-types';

export interface LearningStore {
  // ---- Fingerprints (L1) ----

  /** Get all fingerprints matching a domain and/or task hash */
  getFingerprints(filter: {
    domain: string;
    taskHash?: string;
  }): Promise<LearnedTargetFingerprint[]>;

  /** Store/update fingerprints (merge by key fields) */
  putFingerprints(fingerprints: LearnedTargetFingerprint[]): Promise<void>;

  /** Remove fingerprints older than maxAgeMs */
  pruneFingerprints(maxAgeMs: number): Promise<number>;

  // ---- Domain Profiles (L3) ----

  /** Get the domain profile for a given domain */
  getDomainProfile(domain: string): Promise<DomainProfile | null>;

  /** Store/update a domain profile */
  putDomainProfile(profile: DomainProfile): Promise<void>;

  // ---- General ----

  /** Clear all learned data */
  clear(): Promise<void>;
}

/**
 * In-memory LearningStore implementation for testing and non-extension contexts.
 */
export class InMemoryLearningStore implements LearningStore {
  private fingerprints: LearnedTargetFingerprint[] = [];
  private domainProfiles: Map<string, DomainProfile> = new Map();

  async getFingerprints(filter: {
    domain: string;
    taskHash?: string;
  }): Promise<LearnedTargetFingerprint[]> {
    return this.fingerprints.filter(fp => {
      if (fp.domain !== filter.domain) return false;
      if (filter.taskHash && fp.taskHash !== filter.taskHash) return false;
      return true;
    });
  }

  async putFingerprints(fingerprints: LearnedTargetFingerprint[]): Promise<void> {
    for (const fp of fingerprints) {
      const idx = this.fingerprints.findIndex(
        existing =>
          existing.domain === fp.domain &&
          existing.taskHash === fp.taskHash &&
          existing.intent === fp.intent &&
          (existing.role || '') === (fp.role || '') &&
          existing.hrefPathPattern === fp.hrefPathPattern
      );
      if (idx >= 0) {
        this.fingerprints[idx] = fp;
      } else {
        this.fingerprints.push(fp);
      }
    }
  }

  async pruneFingerprints(maxAgeMs: number): Promise<number> {
    const cutoff = Date.now() - maxAgeMs;
    const before = this.fingerprints.length;
    this.fingerprints = this.fingerprints.filter(fp => fp.learnedAt >= cutoff);
    return before - this.fingerprints.length;
  }

  async getDomainProfile(domain: string): Promise<DomainProfile | null> {
    return this.domainProfiles.get(domain) ?? null;
  }

  async putDomainProfile(profile: DomainProfile): Promise<void> {
    this.domainProfiles.set(profile.domain, profile);
  }

  async clear(): Promise<void> {
    this.fingerprints = [];
    this.domainProfiles.clear();
  }
}
