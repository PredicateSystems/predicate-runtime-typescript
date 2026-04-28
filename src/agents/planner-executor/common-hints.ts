import { HeuristicHint } from './heuristic-hint';

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

export function getCommonHint(intent: string): HeuristicHint | null {
  const normalized = intent.toLowerCase().replace(/[\s-]+/g, '_');
  const exactMatch = COMMON_HINTS[normalized as keyof typeof COMMON_HINTS];
  if (exactMatch) {
    return exactMatch;
  }

  for (const [key, hint] of Object.entries(COMMON_HINTS)) {
    if (normalized.includes(key) || key.includes(normalized)) {
      return hint;
    }
  }

  return null;
}
