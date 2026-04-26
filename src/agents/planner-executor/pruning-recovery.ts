import type { Snapshot, SnapshotElement } from './plan-models';
import { pruneSnapshotForTask } from './category-pruner';
import type { PrunedSnapshotContext, PruningRecoveryOptions } from './pruning-types';

export const DEFAULT_MIN_PRUNED_ELEMENTS = 3;
const INTENT_STOP_WORDS = new Set([
  'button',
  'link',
  'field',
  'input',
  'open',
  'click',
  'tap',
  'press',
  'select',
  'choose',
  'find',
  'locate',
  'control',
  'element',
  'item',
  'the',
  'this',
  'that',
  'these',
  'those',
  'into',
  'from',
  'with',
  'for',
]);

function normalizeIntentTokens(intent: string): string[] {
  return intent
    .toLowerCase()
    .replace(/[_-]/g, ' ')
    .split(/\s+/)
    .filter(token => token.length > 2 && !INTENT_STOP_WORDS.has(token));
}

function elementMatchesIntent(element: SnapshotElement, intent: string): boolean {
  const haystack = [element.text, element.ariaLabel, element.name]
    .filter((value): value is string => typeof value === 'string' && value.trim().length > 0)
    .join(' ')
    .toLowerCase();
  const tokens = normalizeIntentTokens(intent);
  if (tokens.length === 0) {
    return false;
  }
  return tokens.every(token => haystack.includes(token));
}

export function fullSnapshotContainsIntent(snapshot: Snapshot | null, intent?: string): boolean {
  if (!snapshot || !intent) {
    return false;
  }
  return (snapshot.elements || []).some(element => elementMatchesIntent(element, intent));
}

export function pruneWithRecovery(
  snapshot: Snapshot,
  options: PruningRecoveryOptions
): PrunedSnapshotContext {
  const minElementCount = Math.max(1, options.minElementCount ?? DEFAULT_MIN_PRUNED_ELEMENTS);
  const maxRelaxation = Math.max(options.relaxationLevel || 0, options.maxRelaxation ?? 2);

  let current = pruneSnapshotForTask(snapshot, options);
  for (let level = current.relaxationLevel + 1; level <= maxRelaxation; level += 1) {
    if (current.actionableElementCount >= minElementCount) {
      break;
    }
    current = pruneSnapshotForTask(snapshot, { ...options, relaxationLevel: level });
  }

  return current;
}
