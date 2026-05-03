import type { Snapshot, SnapshotElement } from './plan-models';
import type { DataDrivenPruningPolicy, LearnedTargetFingerprint } from './profile-types';

export enum PruningTaskCategory {
  SHOPPING = 'shopping',
  FORM_FILLING = 'form_filling',
  SEARCH = 'search',
  CHECKOUT = 'checkout',
  GENERIC = 'generic',
}

export interface PrunedSnapshotContext {
  category: PruningTaskCategory;
  snapshot: Snapshot;
  elements: SnapshotElement[];
  promptBlock: string;
  relaxationLevel: number;
  rawElementCount: number;
  prunedElementCount: number;
  actionableElementCount: number;
}

export interface PruneSnapshotOptions {
  goal: string;
  category: PruningTaskCategory;
  relaxationLevel?: number;
  /** Optional data-driven pruning policy from a resolved BrowserAgentProfile */
  profilePolicy?: DataDrivenPruningPolicy;
  /** Optional learned fingerprints for fingerprint-boost scoring */
  learnedFingerprints?: LearnedTargetFingerprint[];
}

export interface PruningRecoveryOptions extends PruneSnapshotOptions {
  maxRelaxation?: number;
  minElementCount?: number;
}
