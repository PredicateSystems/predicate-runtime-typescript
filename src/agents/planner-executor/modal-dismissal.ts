/**
 * Modal/Overlay Dismissal Logic
 *
 * Handles automatic dismissal of blocking overlays after DOM changes:
 * - Product protection/warranty upsells
 * - Cookie consent banners
 * - Newsletter signup popups
 * - Promotional overlays
 * - Cart upsell drawers
 *
 * Uses word boundary matching to avoid false positives.
 */

import type { SnapshotElement } from './plan-models';

/**
 * Configuration for modal dismissal.
 */
export interface ModalDismissalConfig {
  /** Whether modal dismissal is enabled (default: true) */
  enabled: boolean;
  /** Maximum dismissal attempts per modal (default: 2) */
  maxAttempts: number;
  /** Minimum new elements to consider as modal appearance (default: 5) */
  minNewElements: number;
  /** Element roles to consider for dismissal (default: ['button', 'link']) */
  roleFilter: string[];
  /** Dismissal text patterns (decline/skip, close, continue) */
  dismissPatterns: string[];
  /** Icon patterns for close buttons (exact match) */
  iconPatterns: string[];
  /** Checkout button patterns to skip dismissal when found */
  checkoutPatterns: string[];
}

/**
 * Default modal dismissal configuration.
 */
export const DEFAULT_MODAL_CONFIG: ModalDismissalConfig = {
  enabled: true,
  maxAttempts: 2,
  minNewElements: 5,
  roleFilter: ['button', 'link'],
  dismissPatterns: [
    // Decline/skip patterns (highest priority)
    'no thanks',
    'no, thanks',
    'not now',
    'skip',
    'decline',
    'maybe later',
    'not interested',
    // Close patterns
    'close',
    'dismiss',
    'cancel',
    'x',
    // Continue patterns (lower priority)
    'continue',
    'proceed',
    'ok',
    'got it',
    'i understand',
  ],
  iconPatterns: ['x', '×', '✕', '✖', '✗', '╳'],
  checkoutPatterns: [
    'checkout',
    'check out',
    'proceed to checkout',
    'go to checkout',
    'view cart',
    'view bag',
    'shopping cart',
    'shopping bag',
    'continue to checkout',
    'secure checkout',
    'go to cart',
    'see cart',
    'go to bag',
  ],
};

/**
 * Candidate element for modal dismissal.
 */
interface DismissCandidate {
  /** Element ID */
  id: number;
  /** Match score (higher = better) */
  score: number;
  /** Pattern that matched */
  matchedPattern: string;
}

/**
 * Result of modal dismissal candidate search.
 */
export interface ModalDismissalResult {
  /** Whether a dismissal target was found */
  found: boolean;
  /** Element ID to click for dismissal */
  elementId: number | null;
  /** The pattern that matched */
  matchedPattern: string | null;
  /** Whether checkout button was detected (skip dismissal) */
  hasCheckoutButton: boolean;
}

/**
 * Check if text matches a pattern using word boundary matching.
 *
 * This avoids false positives like:
 * - "mexico" matching "x"
 * - "enclosed" matching "close"
 * - "boxer" matching "x"
 *
 * @param text - Text to search in
 * @param pattern - Pattern to match
 * @returns true if pattern matches with word boundaries
 */
function wordBoundaryMatch(text: string, pattern: string): boolean {
  const textLower = text.toLowerCase();
  const patternLower = pattern.toLowerCase();

  // For single-character patterns like "x", require exact match
  if (patternLower.length === 1) {
    return textLower === patternLower;
  }

  // For longer patterns, use word boundary regex
  try {
    const regex = new RegExp(`\\b${escapeRegex(patternLower)}\\b`, 'i');
    return regex.test(textLower);
  } catch {
    // Fallback to simple includes if regex fails
    return textLower.includes(patternLower);
  }
}

/**
 * Escape special regex characters.
 */
function escapeRegex(str: string): string {
  return str.replace(/[.*+?^${}()|[\]\\]/g, '\\$&');
}

/**
 * Check if element is a global navigation cart link (skip these).
 *
 * @param element - Element to check
 * @returns true if element appears to be global nav
 */
function isGlobalNavCartLink(element: SnapshotElement): boolean {
  const text = (element.text || '').toLowerCase();
  const ariaLabel = (element.ariaLabel || '').toLowerCase();
  const href = (element.href || '').toLowerCase();

  // Skip if it's a small cart count indicator
  if (/^\d+$/.test(text.trim())) {
    return true;
  }

  // Skip if it looks like main nav cart
  if (text === 'cart' && href.includes('/cart')) {
    // Check if there's additional context suggesting it's not an overlay button
    return true;
  }

  return false;
}

/**
 * Find the best element to dismiss a modal/overlay.
 *
 * Looks for buttons with common dismissal text patterns
 * and returns the best candidate.
 *
 * CRITICAL: First checks if the overlay contains clickable checkout-related
 * elements. If found, skips dismissal since the user should interact with those.
 *
 * @param elements - Elements from post-action snapshot
 * @param config - Modal dismissal configuration
 * @returns Modal dismissal result with element ID if found
 *
 * @example
 * ```typescript
 * const result = findDismissalTarget(elements, config);
 * if (result.found && result.elementId !== null) {
 *   await runtime.click(result.elementId);
 * }
 * ```
 */
export function findDismissalTarget(
  elements: SnapshotElement[],
  config: ModalDismissalConfig = DEFAULT_MODAL_CONFIG
): ModalDismissalResult {
  if (!config.enabled) {
    return {
      found: false,
      elementId: null,
      matchedPattern: null,
      hasCheckoutButton: false,
    };
  }

  // CRITICAL: Check for clickable checkout buttons first
  // Only skip if there's an actual button/link (not just text)
  for (const element of elements) {
    const role = (element.role || '').toLowerCase();

    // Only consider buttons and links
    if (!['button', 'link'].includes(role)) {
      continue;
    }

    // Skip global nav cart links
    if (isGlobalNavCartLink(element)) {
      continue;
    }

    const text = (element.text || '').toLowerCase();
    const ariaLabel = (element.ariaLabel || '').toLowerCase();
    const href = (element.href || '').toLowerCase();

    // Check text/aria-label for checkout patterns
    for (const pattern of config.checkoutPatterns) {
      if (text.includes(pattern) || ariaLabel.includes(pattern)) {
        return {
          found: false,
          elementId: null,
          matchedPattern: null,
          hasCheckoutButton: true,
        };
      }
    }

    // Check href for cart/checkout links
    if (href.includes('cart') || href.includes('checkout') || href.includes('bag')) {
      return {
        found: false,
        elementId: null,
        matchedPattern: null,
        hasCheckoutButton: true,
      };
    }
  }

  // Find candidates that match dismissal patterns
  const candidates: DismissCandidate[] = [];

  for (const element of elements) {
    const id = element.id;
    if (id === undefined) continue;

    const role = (element.role || '').toLowerCase();

    // Only consider specified roles
    if (!config.roleFilter.includes(role)) {
      continue;
    }

    const text = (element.text || '').toLowerCase().trim();
    const ariaLabel = (element.ariaLabel || '').toLowerCase();

    // Check for icon patterns (exact match, highest priority)
    for (const icon of config.iconPatterns) {
      if (text === icon || ariaLabel === icon) {
        candidates.push({
          id,
          score: 200, // Highest priority
          matchedPattern: icon,
        });
        break;
      }
    }

    // Check for dismissal patterns (word boundary match)
    for (let i = 0; i < config.dismissPatterns.length; i++) {
      const pattern = config.dismissPatterns[i];

      if (wordBoundaryMatch(text, pattern) || wordBoundaryMatch(ariaLabel, pattern)) {
        // Score: earlier patterns have higher priority
        candidates.push({
          id,
          score: 100 - i,
          matchedPattern: pattern,
        });
        break;
      }
    }
  }

  // Return best candidate (highest score)
  if (candidates.length === 0) {
    return {
      found: false,
      elementId: null,
      matchedPattern: null,
      hasCheckoutButton: false,
    };
  }

  candidates.sort((a, b) => b.score - a.score);
  const best = candidates[0];

  return {
    found: true,
    elementId: best.id,
    matchedPattern: best.matchedPattern,
    hasCheckoutButton: false,
  };
}

/**
 * Detect if significant DOM change occurred (potential modal).
 *
 * @param preElements - Element IDs before action
 * @param postElements - Element IDs after action
 * @param minNewElements - Minimum new elements to consider as modal
 * @returns true if modal-like change detected
 */
export function detectModalAppearance(
  preElements: Set<number>,
  postElements: Set<number>,
  minNewElements: number = DEFAULT_MODAL_CONFIG.minNewElements
): boolean {
  const newElements = new Set<number>();
  for (const id of postElements) {
    if (!preElements.has(id)) {
      newElements.add(id);
    }
  }
  return newElements.size >= minNewElements;
}

/**
 * Detect if modal was successfully dismissed.
 *
 * @param preElements - Element IDs before dismissal attempt
 * @param postElements - Element IDs after dismissal attempt
 * @param minRemovedElements - Minimum removed elements to consider dismissed (default: 3)
 * @returns true if significant elements were removed (modal dismissed)
 */
export function detectModalDismissed(
  preElements: Set<number>,
  postElements: Set<number>,
  minRemovedElements: number = 3
): boolean {
  const removedElements = new Set<number>();
  for (const id of preElements) {
    if (!postElements.has(id)) {
      removedElements.add(id);
    }
  }
  return removedElements.size >= minRemovedElements;
}
