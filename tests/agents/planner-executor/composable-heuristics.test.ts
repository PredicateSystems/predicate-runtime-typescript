import type { SnapshotElement } from '../../../src/agents/planner-executor/plan-models';
import type { IntentHeuristics } from '../../../src/agents/planner-executor/planner-executor-agent';
import { TaskCategory } from '../../../src/agents/planner-executor/task-category';
import { HeuristicHint } from '../../../src/agents/planner-executor/heuristic-hint';
import { COMMON_HINTS, getCommonHint } from '../../../src/agents/planner-executor/common-hints';
import { ComposableHeuristics } from '../../../src/agents/planner-executor/composable-heuristics';

function makeElement(
  id: number,
  overrides: Partial<SnapshotElement> = {}
): SnapshotElement & { attributes?: Record<string, string> } {
  return {
    id,
    role: 'button',
    text: '',
    clickable: true,
    ...overrides,
  };
}

class StaticHeuristicsStub implements IntentHeuristics {
  constructor(
    private readonly result: number | null,
    private readonly patterns: string[] = ['static']
  ) {}

  findElementForIntent(): number | null {
    return this.result;
  }

  priorityOrder(): string[] {
    return this.patterns;
  }
}

describe('HeuristicHint', () => {
  it('matches intent patterns case-insensitively', () => {
    const hint = new HeuristicHint({
      intentPattern: 'add_to_cart',
      textPatterns: ['add to cart'],
      roleFilter: ['button'],
      priority: 10,
    });

    expect(hint.matchesIntent('Add_To_Cart')).toBe(true);
    expect(hint.matchesIntent('checkout')).toBe(false);
  });

  it('normalizes separators when matching planner intents', () => {
    const hint = new HeuristicHint({
      intentPattern: 'product_link',
      textPatterns: ['wireless earbuds'],
      roleFilter: ['link'],
      priority: 10,
    });

    expect(hint.matchesIntent('product link')).toBe(true);
    expect(hint.matchesIntent('product-link')).toBe(true);
    expect(hint.matchesIntent('PRODUCT_LINK')).toBe(true);
  });

  it('matches elements by role, text, and attribute patterns', () => {
    const hint = new HeuristicHint({
      intentPattern: 'product_link',
      textPatterns: ['vinyl tablecloth'],
      roleFilter: ['link'],
      priority: 9,
      attributePatterns: { href: '/products/' },
    });

    const matching = makeElement(1, {
      role: 'link',
      text: 'Round Vinyl Tablecloth',
      href: '/products/vinyl-tablecloth',
    });
    const wrongRole = makeElement(2, {
      role: 'button',
      text: 'Round Vinyl Tablecloth',
      href: '/products/vinyl-tablecloth',
    });

    expect(hint.matchesElement(matching)).toBe(true);
    expect(hint.matchesElement(wrongRole)).toBe(false);
  });
});

describe('COMMON_HINTS', () => {
  it('contains the well-known Python parity intents', () => {
    expect(COMMON_HINTS.add_to_cart.intentPattern).toBe('add_to_cart');
    expect(COMMON_HINTS.checkout.intentPattern).toBe('checkout');
    expect(COMMON_HINTS.login.intentPattern).toBe('login');
    expect(COMMON_HINTS.submit.intentPattern).toBe('submit');
    expect(COMMON_HINTS.search.intentPattern).toBe('search');
    expect(COMMON_HINTS.close.intentPattern).toBe('close');
    expect(COMMON_HINTS.accept_cookies.intentPattern).toBe('accept_cookies');
  });

  it('normalizes common hint lookups', () => {
    expect(getCommonHint('add-to-cart')?.intentPattern).toBe('add_to_cart');
    expect(getCommonHint('accept cookies')?.intentPattern).toBe('accept_cookies');
    expect(getCommonHint('unknown')).toBeNull();
  });
});

describe('ComposableHeuristics', () => {
  it('prefers planner step hints over common hints and static heuristics', () => {
    const heuristics = new ComposableHeuristics({
      staticHeuristics: new StaticHeuristicsStub(99, ['static_submit']),
      taskCategory: TaskCategory.TRANSACTION,
    });
    heuristics.setStepHints([
      {
        intent_pattern: 'submit',
        text_patterns: ['special submit'],
        role_filter: ['button'],
        priority: 20,
      },
    ]);

    const elements = [
      makeElement(1, { text: 'Special Submit', role: 'button' }),
      makeElement(2, { text: 'Submit', role: 'button' }),
    ];

    expect(
      heuristics.findElementForIntent('submit', elements, 'https://example.com', 'Submit form')
    ).toBe(1);
  });

  it('falls back to common hints before static heuristics', () => {
    const heuristics = new ComposableHeuristics({
      staticHeuristics: new StaticHeuristicsStub(99, ['custom_checkout']),
      taskCategory: TaskCategory.TRANSACTION,
    });

    const elements = [makeElement(10, { text: 'Proceed to Checkout', role: 'button' })];

    expect(
      heuristics.findElementForIntent('checkout', elements, 'https://shop.test/cart', 'Checkout')
    ).toBe(10);
  });

  it('treats searchbox and combobox fields as first-class search inputs', () => {
    const heuristics = new ComposableHeuristics({
      staticHeuristics: new StaticHeuristicsStub(null),
      taskCategory: TaskCategory.SEARCH,
    });

    const searchboxElements = [
      makeElement(12, { text: 'Search Amazon', role: 'searchbox' }),
      makeElement(13, { text: 'Sign in', role: 'button' }),
    ];
    const comboboxElements = [
      makeElement(21, { text: 'Search products', role: 'combobox' }),
      makeElement(22, { text: 'Search', role: 'button' }),
    ];

    expect(
      heuristics.findElementForIntent(
        'searchbox',
        searchboxElements,
        'https://shop.test',
        'Search for headphones'
      )
    ).toBe(12);
    expect(
      heuristics.findElementForIntent(
        'searchbox',
        comboboxElements,
        'https://shop.test',
        'Search for headphones'
      )
    ).toBe(21);
  });

  it('matches a searchbox role even when the visible text is an implementation name', () => {
    const heuristics = new ComposableHeuristics({
      staticHeuristics: new StaticHeuristicsStub(null),
      taskCategory: TaskCategory.SEARCH,
    });

    const elements = [
      makeElement(31, { text: 'field-keywords', role: 'searchbox' }),
      makeElement(32, { text: 'Sign in', role: 'button' }),
    ];

    expect(
      heuristics.findElementForIntent(
        'searchbox',
        elements,
        'https://www.amazon.com/',
        'Search for noise canceling earbuds'
      )
    ).toBe(31);
  });

  it('matches Amazon product card intents to real product detail links', () => {
    const heuristics = new ComposableHeuristics({
      staticHeuristics: new StaticHeuristicsStub(null),
      taskCategory: TaskCategory.SEARCH,
    });

    const elements = [
      makeElement(41, {
        text: 'Sponsored banner',
        role: 'link',
        href: '/gp/help/customer/display.html',
      }),
      makeElement(42, {
        text: 'Wireless Noise Canceling Earbuds with Charging Case',
        role: 'link',
        href: '/dp/B0TEST1234/ref=sr_1_1',
      }),
    ];

    expect(
      heuristics.findElementForIntent(
        'product_card',
        elements,
        'https://www.amazon.com/s?k=noise+canceling+earbuds',
        'Pick an earbuds product'
      )
    ).toBe(42);
  });

  it('falls back to static heuristics before task-category defaults', () => {
    const heuristics = new ComposableHeuristics({
      staticHeuristics: new StaticHeuristicsStub(42, ['custom_fallback']),
      taskCategory: TaskCategory.TRANSACTION,
    });

    const elements = [makeElement(7, { text: 'Add to Cart', role: 'button' })];

    expect(
      heuristics.findElementForIntent(
        'unknown_intent',
        elements,
        'https://shop.test/product',
        'Unknown flow'
      )
    ).toBe(42);
  });

  it('uses task-category defaults last', () => {
    const heuristics = new ComposableHeuristics({
      staticHeuristics: new StaticHeuristicsStub(null),
      taskCategory: TaskCategory.TRANSACTION,
    });

    const elements = [
      makeElement(5, { text: 'Place Order', role: 'button' }),
      makeElement(6, { text: 'Privacy Policy', role: 'link' }),
    ];

    expect(
      heuristics.findElementForIntent(
        'complete purchase',
        elements,
        'https://shop.test/checkout',
        'Finish checkout'
      )
    ).toBe(5);
  });

  it('returns a deduplicated priority order across sources', () => {
    const heuristics = new ComposableHeuristics({
      staticHeuristics: new StaticHeuristicsStub(null, ['checkout', 'custom_fallback']),
      taskCategory: TaskCategory.TRANSACTION,
    });
    heuristics.setStepHints([
      new HeuristicHint({
        intentPattern: 'submit',
        textPatterns: ['submit'],
        roleFilter: ['button'],
        priority: 7,
      }),
    ]);

    expect(heuristics.priorityOrder()).toEqual(
      expect.arrayContaining(['submit', 'add_to_cart', 'checkout', 'custom_fallback'])
    );
    expect(heuristics.priorityOrder().filter((p: string) => p === 'checkout')).toHaveLength(1);
  });
});
