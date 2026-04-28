/**
 * Boundary Detection for Authentication and Checkout Pages
 *
 * Detects when the agent reaches terminal states that require special handling:
 * - Authentication boundaries (login/sign-in pages)
 * - Checkout pages (may require different handling)
 *
 * Authentication boundaries are graceful terminal states - the agent has
 * successfully navigated as far as possible without credentials.
 */

import type { SnapshotElement, PredicateSpec } from './plan-models';

/**
 * Configuration for authentication boundary detection.
 */
export interface AuthBoundaryConfig {
  /** Whether auth boundary detection is enabled (default: true) */
  enabled: boolean;
  /** URL patterns indicating authentication pages */
  urlPatterns: string[];
  /** If true, mark run as successful when auth boundary reached (default: true) */
  stopOnAuth: boolean;
  /** Message to include in outcome when stopping at auth (default: "Reached authentication boundary (login required)") */
  authSuccessMessage: string;
}

/**
 * Default auth boundary configuration.
 */
export const DEFAULT_AUTH_BOUNDARY_CONFIG: AuthBoundaryConfig = {
  enabled: true,
  urlPatterns: [
    '/signin',
    '/sign-in',
    '/login',
    '/log-in',
    '/auth',
    '/authenticate',
    '/ap/signin', // Amazon sign-in
    '/ap/register', // Amazon registration
    '/ax/claim', // Amazon CAPTCHA/verification
    '/account/login',
    '/accounts/login',
    '/user/login',
  ],
  stopOnAuth: true,
  authSuccessMessage: 'Reached authentication boundary (login required)',
};

/**
 * Configuration for checkout page detection.
 */
export interface CheckoutDetectionConfig {
  /** Whether checkout detection is enabled (default: true) */
  enabled: boolean;
  /** URL patterns indicating cart pages */
  cartUrlPatterns: string[];
  /** URL patterns indicating checkout pages */
  checkoutUrlPatterns: string[];
  /** Element text patterns indicating checkout-related buttons */
  checkoutElementPatterns: string[];
}

/**
 * Default checkout detection configuration.
 */
export const DEFAULT_CHECKOUT_CONFIG: CheckoutDetectionConfig = {
  enabled: true,
  cartUrlPatterns: ['/cart', '/basket', '/bag', '/shopping-cart', '/gp/cart'],
  checkoutUrlPatterns: ['/checkout', '/buy', '/order', '/payment', '/purchase', '/gp/checkout'],
  checkoutElementPatterns: [
    'proceed to checkout',
    'go to checkout',
    'view cart',
    'shopping cart',
    'your cart',
    'sign in to checkout',
    'continue to payment',
    'place your order',
    'buy now',
  ],
};

/**
 * Result of auth boundary detection.
 */
export interface AuthBoundaryResult {
  /** Whether an auth boundary was detected */
  isAuthBoundary: boolean;
  /** The URL pattern that matched (if any) */
  matchedPattern: string | null;
}

/**
 * Result of checkout page detection.
 */
export interface CheckoutDetectionResult {
  /** Whether a checkout-related page was detected */
  isCheckoutRelated: boolean;
  /** Whether it's a cart page */
  isCart: boolean;
  /** Whether it's a checkout page */
  isCheckout: boolean;
  /** The URL pattern that matched (if any) */
  matchedPattern: string | null;
}

/**
 * Detect if the current URL is an authentication boundary.
 *
 * An auth boundary is a login/sign-in page where the agent cannot
 * proceed without credentials. This is a terminal state.
 *
 * @param url - Current page URL
 * @param config - Auth boundary configuration
 * @returns Auth boundary detection result
 *
 * @example
 * ```typescript
 * const result = detectAuthBoundary('https://amazon.com/ap/signin', config);
 * if (result.isAuthBoundary) {
 *   console.log(`Auth page detected: ${result.matchedPattern}`);
 * }
 * ```
 */
export function detectAuthBoundary(
  url: string,
  config: AuthBoundaryConfig = DEFAULT_AUTH_BOUNDARY_CONFIG
): AuthBoundaryResult {
  if (!config.enabled || !url) {
    return { isAuthBoundary: false, matchedPattern: null };
  }

  const urlLower = url.toLowerCase();

  for (const pattern of config.urlPatterns) {
    if (urlLower.includes(pattern.toLowerCase())) {
      return { isAuthBoundary: true, matchedPattern: pattern };
    }
  }

  return { isAuthBoundary: false, matchedPattern: null };
}

/**
 * Detect if the current URL is a checkout-related page.
 *
 * @param url - Current page URL
 * @param config - Checkout detection configuration
 * @returns Checkout detection result
 *
 * @example
 * ```typescript
 * const result = detectCheckoutPage('https://shop.com/checkout', config);
 * if (result.isCheckout) {
 *   console.log('On checkout page');
 * }
 * ```
 */
export function detectCheckoutPage(
  url: string,
  config: CheckoutDetectionConfig = DEFAULT_CHECKOUT_CONFIG
): CheckoutDetectionResult {
  if (!config.enabled || !url) {
    return {
      isCheckoutRelated: false,
      isCart: false,
      isCheckout: false,
      matchedPattern: null,
    };
  }

  const urlLower = url.toLowerCase();

  // Check cart patterns
  for (const pattern of config.cartUrlPatterns) {
    if (urlLower.includes(pattern.toLowerCase())) {
      return {
        isCheckoutRelated: true,
        isCart: true,
        isCheckout: false,
        matchedPattern: pattern,
      };
    }
  }

  // Check checkout patterns
  for (const pattern of config.checkoutUrlPatterns) {
    if (urlLower.includes(pattern.toLowerCase())) {
      return {
        isCheckoutRelated: true,
        isCart: false,
        isCheckout: true,
        matchedPattern: pattern,
      };
    }
  }

  return {
    isCheckoutRelated: false,
    isCart: false,
    isCheckout: false,
    matchedPattern: null,
  };
}

/**
 * Check if an element text matches checkout-related patterns.
 *
 * @param text - Element text to check
 * @param config - Checkout detection configuration
 * @returns true if text matches a checkout pattern
 */
export function isCheckoutElement(
  text: string,
  config: CheckoutDetectionConfig = DEFAULT_CHECKOUT_CONFIG
): boolean {
  if (!text) return false;

  const textLower = text.toLowerCase();

  for (const pattern of config.checkoutElementPatterns) {
    if (textLower.includes(pattern)) {
      return true;
    }
  }

  return false;
}

function normalizeIntentText(value: string | undefined | null): string {
  return (value || '').toLowerCase().replace(/[_-]+/g, ' ').replace(/\s+/g, ' ').trim();
}

function extractUrlSignals(url: string): string[] {
  if (!url) {
    return [];
  }

  try {
    const parsed = new URL(url);
    const signals = [
      parsed.pathname,
      parsed.search,
      parsed.searchParams.get('q') || '',
      parsed.searchParams.get('query') || '',
      parsed.searchParams.get('search') || '',
      parsed.searchParams.get('keyword') || '',
      parsed.searchParams.get('keywords') || '',
      parsed.searchParams.get('term') || '',
      parsed.searchParams.get('s') || '',
    ];
    return signals.map(signal => normalizeIntentText(signal)).filter(Boolean);
  } catch {
    return [normalizeIntentText(url)];
  }
}

function queryTerms(text: string | undefined | null): string[] {
  return normalizeIntentText(text)
    .split(/\s+/)
    .filter(term => term.length >= 3);
}

function urlPredicateSignals(verify: PredicateSpec[] | undefined): string[] {
  if (!Array.isArray(verify)) {
    return [];
  }

  const signals: string[] = [];
  for (const predicate of verify) {
    if (
      predicate &&
      typeof predicate.predicate === 'string' &&
      (predicate.predicate === 'url_contains' || predicate.predicate === 'url_matches')
    ) {
      const firstArg = predicate.args?.[0];
      if (typeof firstArg === 'string' && firstArg.trim()) {
        signals.push(normalizeIntentText(firstArg));
      }
    }
  }

  return signals;
}

export function isSearchLikeTypeAndSubmit(
  step: { action?: string; intent?: string; input?: string; verify?: PredicateSpec[] },
  element?: Pick<SnapshotElement, 'role' | 'text' | 'name' | 'ariaLabel'> | null
): boolean {
  const action = (step.action || '').toUpperCase();
  if (action !== 'TYPE_AND_SUBMIT' && action !== 'TYPE') {
    return false;
  }

  const cues = [
    normalizeIntentText(step.intent),
    normalizeIntentText(step.input),
    normalizeIntentText(element?.role),
    normalizeIntentText(element?.text),
    normalizeIntentText(element?.name),
    normalizeIntentText(element?.ariaLabel),
    ...urlPredicateSignals(step.verify),
  ].filter(Boolean);

  return cues.some(cue =>
    /\b(search|searchbox|find|lookup|look up|query|keywords?|results?)\b/.test(cue)
  );
}

export function isUrlChangeRelevantToIntent(
  previousUrl: string,
  nextUrl: string,
  step: { action?: string; intent?: string; input?: string; verify?: PredicateSpec[] },
  element?: Pick<SnapshotElement, 'role' | 'text' | 'name' | 'ariaLabel'> | null
): boolean {
  const normalizedPrevious = normalizeIntentText(previousUrl).replace(/\/+$/, '');
  const normalizedNext = normalizeIntentText(nextUrl).replace(/\/+$/, '');
  if (!normalizedNext || normalizedNext === normalizedPrevious) {
    return false;
  }

  const predicateSignals = urlPredicateSignals(step.verify);
  const nextSignals = extractUrlSignals(nextUrl);
  if (
    predicateSignals.length > 0 &&
    predicateSignals.every(signal => nextSignals.some(nextSignal => nextSignal.includes(signal)))
  ) {
    return true;
  }

  if (!isSearchLikeTypeAndSubmit(step, element)) {
    return true;
  }

  const searchTerms = [
    ...queryTerms(step.input),
    ...queryTerms(step.intent),
    'search',
    'query',
    'results',
    'find',
  ];

  return searchTerms.some(term => nextSignals.some(signal => signal.includes(term)));
}
