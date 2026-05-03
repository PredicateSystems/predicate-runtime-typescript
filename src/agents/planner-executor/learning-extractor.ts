/**
 * LearningExtractor — Extracts learned fingerprints from successful step outcomes.
 *
 * Converts structured step outcomes + snapshot elements into normalized
 * LearnedTargetFingerprint entries suitable for persistent storage.
 *
 * Privacy safeguards:
 * - Only extracts from policy-allowed, non-sensitive actions
 * - Uses normalized tokens, never raw text
 * - Applies redaction rules consistent with Gateway upload
 *
 * @see docs/plans/browser-agent/2026-05-02-extensible-categories-and-task-learning.md
 */

import type { SnapshotElement, StepOutcome } from './plan-models';
import { StepStatus } from './plan-models';
import type { LearnedTargetFingerprint } from './profile-types';
import { createFingerprint } from './fingerprint-normalizer';

/** Sensitive URL patterns — learning is disabled on these pages */
const SENSITIVE_URL_PATTERNS = [
  /\/checkout/i,
  /\/payment/i,
  /\/account\/settings/i,
  /\/password/i,
  /\/billing/i,
  /\/security/i,
  /\/login/i,
  /\/signin/i,
  /\/auth/i,
];

/** Sensitive element roles — never learn from these */
const SENSITIVE_ROLES = new Set([
  'password',
  'credit-card-number',
  'credit-card-cvv',
  'credit-card-expiration',
]);

/**
 * Check if a URL is privacy-sensitive (learning should be disabled).
 */
export function isSensitiveUrl(url: string): boolean {
  return SENSITIVE_URL_PATTERNS.some(pattern => pattern.test(url));
}

/**
 * Check if an element involves sensitive data.
 */
function isSensitiveElement(element: SnapshotElement): boolean {
  const role = (element.role || '').toLowerCase();
  if (SENSITIVE_ROLES.has(role)) return true;

  const text =
    `${element.text || ''} ${element.name || ''} ${element.ariaLabel || ''}`.toLowerCase();
  return (
    text.includes('password') ||
    text.includes('credit card') ||
    text.includes('cvv') ||
    text.includes('social security') ||
    text.includes('ssn')
  );
}

/**
 * Extract the element ID from an action string like "CLICK(42)" or "TYPE(12, 'text')".
 */
function extractElementIdFromAction(actionTaken: string | undefined): number | null {
  if (!actionTaken) return null;
  const match = actionTaken.match(/^[A-Z_]+\((\d+)/);
  return match ? parseInt(match[1], 10) : null;
}

/**
 * Extract the intent from the planner action for fingerprinting.
 */
function extractIntentFromOutcome(outcome: StepOutcome): string {
  if (outcome.goal) {
    return outcome.goal
      .toLowerCase()
      .replace(/[\s-]+/g, '_')
      .slice(0, 50);
  }
  if (outcome.actionTaken) {
    const action = outcome.actionTaken.match(/^[A-Z_]+/)?.[0] || 'unknown';
    return action.toLowerCase();
  }
  return 'unknown';
}

export interface LearningExtractionOptions {
  /** Whether learning is enabled (opt-in, default false) */
  learningEnabled: boolean;
  /** The task goal string */
  taskGoal: string;
  /** Current page URL */
  currentUrl: string;
  /** Hash of the task goal for task-scoped matching */
  taskHash: string;
}

/**
 * Result of learning extraction from a single step outcome.
 */
export interface LearningExtractionResult {
  /** Whether extraction was performed */
  extracted: boolean;
  /** The fingerprint (if extracted) */
  fingerprint?: LearnedTargetFingerprint;
  /** Reason extraction was skipped (for audit logging) */
  skipReason?: string;
}

/**
 * Extract a learned fingerprint from a successful step outcome.
 *
 * This is the main entry point for L1 learning. It:
 * 1. Checks if learning is enabled
 * 2. Validates the outcome was successful
 * 3. Checks privacy constraints (sensitive URLs, sensitive elements)
 * 4. Normalizes the element into a fingerprint
 *
 * Returns a result with `extracted: true` and the fingerprint, or
 * `extracted: false` with a skip reason for audit logging.
 */
export function extractFingerprintFromOutcome(
  outcome: StepOutcome,
  snapshotElements: SnapshotElement[] | undefined,
  options: LearningExtractionOptions
): LearningExtractionResult {
  // Check if learning is enabled
  if (!options.learningEnabled) {
    return { extracted: false, skipReason: 'learning_disabled' };
  }

  // Only learn from successful steps
  if (outcome.status !== StepStatus.SUCCESS) {
    return { extracted: false, skipReason: 'step_not_successful' };
  }

  // Check for sensitive URL
  if (isSensitiveUrl(options.currentUrl)) {
    return { extracted: false, skipReason: 'sensitive_url' };
  }

  // We need snapshot elements to find the acted-on element
  if (!snapshotElements || snapshotElements.length === 0) {
    return { extracted: false, skipReason: 'no_snapshot_elements' };
  }

  // Extract element ID from action
  const elementId = extractElementIdFromAction(outcome.actionTaken);
  if (elementId === null) {
    // DONE action or no element — nothing to learn from
    return { extracted: false, skipReason: 'no_element_action' };
  }

  // Find the element in the snapshot
  const element = snapshotElements.find(el => el.id === elementId);
  if (!element) {
    return { extracted: false, skipReason: 'element_not_found' };
  }

  // Check for sensitive element
  if (isSensitiveElement(element)) {
    return { extracted: false, skipReason: 'sensitive_element' };
  }

  // Extract domain from URL
  let domain: string;
  try {
    domain = new URL(options.currentUrl).hostname;
  } catch {
    return { extracted: false, skipReason: 'invalid_url' };
  }

  const intent = extractIntentFromOutcome(outcome);

  const fingerprint = createFingerprint(element, intent, options.taskHash, domain);

  return { extracted: true, fingerprint };
}

/**
 * Apply decay to a fingerprint based on a failure event.
 * Returns an updated fingerprint with incremented failure count and reduced confidence.
 */
export function applyFingerprintFailure(
  fingerprint: LearnedTargetFingerprint
): LearnedTargetFingerprint {
  const failureCount = fingerprint.failureCount + 1;
  const totalAttempts = fingerprint.successCount + failureCount;

  // Confidence decays with failures using exponential decay
  const rawConfidence = fingerprint.successCount / totalAttempts;
  // Apply additional decay factor based on failure count
  const decayFactor = Math.pow(0.8, failureCount);
  const confidence = Math.max(0, Math.min(1, rawConfidence * decayFactor));

  return {
    ...fingerprint,
    failureCount,
    confidence,
  };
}

/**
 * Apply success reinforcement to a fingerprint.
 * Returns an updated fingerprint with incremented success count and updated confidence.
 */
export function applyFingerprintSuccess(
  fingerprint: LearnedTargetFingerprint
): LearnedTargetFingerprint {
  const successCount = fingerprint.successCount + 1;
  const totalAttempts = successCount + fingerprint.failureCount;

  // Confidence is ratio of successes to total attempts, capped at 1
  const confidence = Math.min(1, successCount / totalAttempts);

  return {
    ...fingerprint,
    successCount,
    confidence,
    lastUsedAt: Date.now(),
  };
}

/**
 * Check if a fingerprint should be disabled due to too many failures.
 * Fingerprints with 3+ consecutive failures and confidence < 0.3 are considered stale.
 */
export function isFingerprintStale(fingerprint: LearnedTargetFingerprint): boolean {
  return fingerprint.failureCount >= 3 && fingerprint.confidence < 0.3;
}

/**
 * Check if a fingerprint has expired based on TTL.
 */
export function isFingerprintExpired(
  fingerprint: LearnedTargetFingerprint,
  ttlMs: number
): boolean {
  const age = Date.now() - fingerprint.learnedAt;
  return age > ttlMs;
}

/**
 * Convert a learned fingerprint into a HeuristicHintInput for injection into ComposableHeuristics.
 * Only fingerprints with confidence >= minConfidence are converted.
 */
export function fingerprintToHint(
  fingerprint: LearnedTargetFingerprint,
  minConfidence: number = 0.4
): { intent: string; textPatterns: string[]; roleFilter: string[]; priority: number } | null {
  if (fingerprint.confidence < minConfidence) {
    return null;
  }

  if (isFingerprintStale(fingerprint)) {
    return null;
  }

  const textPatterns: string[] = [];
  if (fingerprint.textTokens) {
    textPatterns.push(...fingerprint.textTokens);
  }
  if (fingerprint.ariaTokens) {
    textPatterns.push(...fingerprint.ariaTokens);
  }

  const roleFilter: string[] = [];
  if (fingerprint.role) {
    roleFilter.push(fingerprint.role);
  }

  return {
    intent: fingerprint.intent,
    textPatterns,
    roleFilter,
    // Priority proportional to confidence (range 1-10)
    priority: Math.round(fingerprint.confidence * 10),
  };
}

/**
 * Compute a task hash from a task goal string.
 * Uses a simple hash for environments where crypto.subtle may not be available.
 * Returns first 16 hex chars of SHA-256 when available, or a fallback hash.
 */
export async function computeTaskHash(taskGoal: string): Promise<string> {
  const normalized = taskGoal.trim().toLowerCase().replace(/\s+/g, ' ');

  // Try Web Crypto API first (available in both extension and browser contexts)
  if (typeof crypto !== 'undefined' && crypto.subtle) {
    try {
      const encoder = new TextEncoder();
      const data = encoder.encode(normalized);
      const hashBuffer = await crypto.subtle.digest('SHA-256', data);
      const hashArray = Array.from(new Uint8Array(hashBuffer));
      const hashHex = hashArray.map(b => b.toString(16).padStart(2, '0')).join('');
      return hashHex.slice(0, 16);
    } catch {
      // Fall through to simple hash
    }
  }

  // Simple fallback hash (djb2 algorithm)
  let hash = 5381;
  for (let i = 0; i < normalized.length; i++) {
    hash = (hash << 5) + hash + normalized.charCodeAt(i);
    hash = hash & hash; // Convert to 32-bit integer
  }
  return Math.abs(hash).toString(16).padStart(16, '0').slice(0, 16);
}
