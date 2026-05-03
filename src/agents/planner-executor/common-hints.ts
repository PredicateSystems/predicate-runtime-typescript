import { HeuristicHint } from './heuristic-hint';
import type { HeuristicHintInput } from './heuristic-hint';

export const COMMON_HINTS = {
  add_to_cart: new HeuristicHint({
    intentPattern: 'add_to_cart',
    textPatterns: ['add to cart', 'add to bag', 'add to basket', 'buy now'],
    roleFilter: ['button'],
    priority: 10,
  }),
  checkout: new HeuristicHint({
    intentPattern: 'checkout',
    textPatterns: ['checkout', 'proceed to checkout', 'go to checkout'],
    roleFilter: ['button', 'link'],
    priority: 10,
  }),
  product_card: new HeuristicHint({
    intentPattern: 'product_card',
    roleFilter: ['link'],
    priority: 8,
    attributePatterns: { href: '/dp/' },
  }),
  login: new HeuristicHint({
    intentPattern: 'login',
    textPatterns: ['log in', 'login', 'sign in', 'signin'],
    roleFilter: ['button', 'link'],
    priority: 10,
  }),
  submit: new HeuristicHint({
    intentPattern: 'submit',
    textPatterns: ['submit', 'send', 'continue', 'next', 'confirm'],
    roleFilter: ['button'],
    priority: 5,
  }),
  search: new HeuristicHint({
    intentPattern: 'search',
    textPatterns: ['search', 'find', 'go'],
    roleFilter: ['button', 'textbox', 'searchbox', 'combobox'],
    priority: 5,
  }),
  searchbox: new HeuristicHint({
    intentPattern: 'searchbox',
    roleFilter: ['searchbox'],
    priority: 9,
  }),
  close: new HeuristicHint({
    intentPattern: 'close',
    textPatterns: ['close', 'dismiss', 'x', 'cancel'],
    roleFilter: ['button'],
    priority: 3,
  }),
  accept_cookies: new HeuristicHint({
    intentPattern: 'accept_cookies',
    textPatterns: ['accept', 'accept all', 'allow', 'agree', 'ok', 'got it'],
    roleFilter: ['button'],
    priority: 8,
  }),
} as const;

/**
 * Look up a heuristic hint by intent string.
 *
 * @param intent - The intent to look up (e.g., "add_to_cart", "book_flight")
 * @param profileHints - Optional profile-provided hints to check after built-in hints
 * @returns Matching HeuristicHint or null
 */
export function getCommonHint(
  intent: string,
  profileHints?: HeuristicHintInput[]
): HeuristicHint | null {
  const normalized = intent.toLowerCase().replace(/[\s-]+/g, '_');

  // Check built-in hints first
  const exactMatch = COMMON_HINTS[normalized as keyof typeof COMMON_HINTS];
  if (exactMatch) {
    return exactMatch;
  }

  for (const [key, hint] of Object.entries(COMMON_HINTS)) {
    if (normalized.includes(key) || key.includes(normalized)) {
      return hint;
    }
  }

  // Check profile-provided hints
  if (profileHints && profileHints.length > 0) {
    for (const ph of profileHints) {
      const pattern = ph.intentPattern ?? ph.intent_pattern ?? '';
      const phNormalized = pattern.toLowerCase().replace(/[\s-]+/g, '_');
      if (normalized.includes(phNormalized) || phNormalized.includes(normalized)) {
        return new HeuristicHint({
          intentPattern: pattern,
          textPatterns: ph.textPatterns ?? ph.text_patterns,
          roleFilter: ph.roleFilter ?? ph.role_filter,
          attributePatterns: ph.attributePatterns ?? ph.attribute_patterns,
          priority: ph.priority,
        });
      }
    }
  }

  return null;
}
