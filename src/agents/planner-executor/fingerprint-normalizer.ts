/**
 * Fingerprint normalizer — creates privacy-safe, stable fingerprints from
 * successful element interactions for L1 learning.
 *
 * Privacy gates:
 * - Text tokens are capped at 4 tokens, each ≤ 20 chars
 * - Sensitive tokens (email, phone, credit card patterns) are redacted
 * - href values are reduced to path patterns only (no query params with PII)
 * - Attribute values are capped at 30 chars
 *
 * @see docs/plans/browser-agent/2026-05-02-extensible-categories-and-task-learning.md
 */

import type { SnapshotElement } from './plan-models';
import type { LearnedTargetFingerprint } from './profile-types';

const MAX_TEXT_TOKENS = 4;
const MAX_TOKEN_LENGTH = 20;
const MAX_ATTR_VALUE_LENGTH = 30;

const SENSITIVE_PATTERNS = [
  /@\S+\.\S+/, // email
  /\d{3}[-.\s]?\d{3}[-.\s]?\d{4}/, // phone
  /\d{4}[-\s]?\d{4}[-\s]?\d{4}[-\s]?\d{4}/, // credit card
  /password/i,
  /ssn/i,
  /social.security/i,
  /date.of.birth/i,
  /dob/i,
  /account.number/i,
  /routing.number/i,
];

function isSensitiveToken(token: string): boolean {
  return SENSITIVE_PATTERNS.some(p => p.test(token));
}

function normalizeTextTokens(text: string | undefined): string[] {
  if (!text) return [];
  const tokens = text
    .split(/\s+/)
    .map(t => t.trim())
    .filter(t => t.length > 0 && t.length <= MAX_TOKEN_LENGTH)
    .filter(t => !isSensitiveToken(t))
    .slice(0, MAX_TEXT_TOKENS);
  return tokens;
}

function normalizeHrefPath(href: string | undefined): string | undefined {
  if (!href) return undefined;
  try {
    const url = new URL(href);
    // Keep only pathname, strip query/hash for privacy
    return url.pathname || undefined;
  } catch {
    // Not a full URL — maybe a relative path
    if (href.startsWith('/')) {
      const queryIndex = href.indexOf('?');
      return queryIndex >= 0 ? href.substring(0, queryIndex) : href;
    }
    return undefined;
  }
}

function normalizeAttributes(element: SnapshotElement): Record<string, string> | undefined {
  const attrs = (element as any).attributes as Record<string, string> | undefined;
  if (!attrs || typeof attrs !== 'object') return undefined;

  const result: Record<string, string> = {};
  for (const [key, value] of Object.entries(attrs)) {
    if (
      typeof value === 'string' &&
      value.length <= MAX_ATTR_VALUE_LENGTH &&
      !isSensitiveToken(value)
    ) {
      result[key] = value;
    }
  }
  return Object.keys(result).length > 0 ? result : undefined;
}

/**
 * Compute a simple hash of the task goal for task-scoped fingerprint matching.
 * Uses a fast, non-cryptographic hash suitable for grouping similar tasks.
 */
export function computeTaskHash(taskGoal: string): string {
  const normalized = taskGoal.toLowerCase().trim().replace(/\s+/g, ' ');
  let hash = 0;
  for (let i = 0; i < normalized.length; i++) {
    const chr = normalized.charCodeAt(i);
    hash = (hash << 5) - hash + chr;
    hash |= 0; // Convert to 32-bit int
  }
  return `th_${Math.abs(hash).toString(36)}`;
}

/**
 * Extract domain from a URL string.
 */
export function extractDomain(url: string): string {
  try {
    return new URL(url).hostname;
  } catch {
    return '';
  }
}

/**
 * Create a learned fingerprint from a successful element interaction.
 */
export function createFingerprint(
  element: SnapshotElement,
  intent: string,
  taskHash: string,
  domain: string
): LearnedTargetFingerprint {
  return {
    domain,
    taskHash,
    intent,
    role: element.role || undefined,
    textTokens: normalizeTextTokens(element.text || element.name),
    ariaTokens: normalizeTextTokens(element.ariaLabel),
    hrefPathPattern: normalizeHrefPath(element.href),
    attributePatterns: normalizeAttributes(element),
    successCount: 1,
    failureCount: 0,
    confidence: 0.5,
    learnedAt: Date.now(),
  };
}

/**
 * Merge a new fingerprint into an existing list, updating counts and confidence.
 * Returns a new array (immutable).
 */
export function mergeFingerprint(
  existing: LearnedTargetFingerprint[],
  newFp: LearnedTargetFingerprint
): LearnedTargetFingerprint[] {
  const matchIdx = existing.findIndex(
    fp =>
      fp.domain === newFp.domain &&
      fp.taskHash === newFp.taskHash &&
      fp.intent === newFp.intent &&
      (fp.role || '') === (newFp.role || '') &&
      fp.hrefPathPattern === newFp.hrefPathPattern
  );

  if (matchIdx >= 0) {
    const existing2 = existing[matchIdx];
    const successCount = existing2.successCount + 1;
    const total = successCount + existing2.failureCount;
    const confidence = Math.min(1, successCount / total);
    const updated: LearnedTargetFingerprint = {
      ...existing2,
      textTokens: newFp.textTokens ?? existing2.textTokens,
      ariaTokens: newFp.ariaTokens ?? existing2.ariaTokens,
      attributePatterns: newFp.attributePatterns ?? existing2.attributePatterns,
      successCount,
      confidence,
      lastUsedAt: Date.now(),
    };
    const result = [...existing];
    result[matchIdx] = updated;
    return result;
  }

  return [...existing, newFp];
}

/**
 * Record a failure for a matching fingerprint (if found).
 */
export function recordFingerprintFailure(
  existing: LearnedTargetFingerprint[],
  intent: string,
  taskHash: string,
  domain: string
): LearnedTargetFingerprint[] {
  const matchIdx = existing.findIndex(
    fp => fp.domain === domain && fp.taskHash === taskHash && fp.intent === intent
  );

  if (matchIdx >= 0) {
    const fp = existing[matchIdx];
    const failureCount = fp.failureCount + 1;
    const total = fp.successCount + failureCount;
    const confidence = Math.max(0, fp.successCount / total);
    const updated: LearnedTargetFingerprint = {
      ...fp,
      failureCount,
      confidence,
    };
    const result = [...existing];
    result[matchIdx] = updated;
    return result;
  }

  return existing;
}
