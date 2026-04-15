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
