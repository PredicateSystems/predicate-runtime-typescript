import {
  PruningTaskCategory,
  pruneSnapshotForTask,
} from '../../../src/agents/planner-executor/category-pruner';
import type { Snapshot, SnapshotElement } from '../../../src/agents/planner-executor/plan-models';

function makeElement(
  overrides: Partial<SnapshotElement> & Pick<SnapshotElement, 'id'>
): SnapshotElement {
  return {
    role: 'button',
    text: '',
    importance: 0,
    clickable: false,
    ...overrides,
  };
}

function makeSnapshot(elements: SnapshotElement[]): Snapshot {
  return {
    url: 'https://shop.test/page',
    title: 'Test Page',
    elements,
  };
}

describe('category-pruner', () => {
  it('produces shopping pruning output with category metadata and relevant elements', () => {
    const snapshot = makeSnapshot([
      makeElement({
        id: 1,
        role: 'link',
        text: 'Rainbow Trout Trucker',
        href: '/product/hat',
        importance: 900,
        clickable: true,
        inDominantGroup: true,
      }),
      makeElement({
        id: 2,
        role: 'text',
        text: '$32.50',
        importance: 850,
        nearbyText: 'Price',
      }),
      makeElement({
        id: 3,
        role: 'button',
        text: 'Add to Cart',
        importance: 950,
        clickable: true,
      }),
      makeElement({
        id: 4,
        role: 'link',
        text: 'Privacy Policy',
        href: '/privacy',
        importance: 50,
        clickable: true,
      }),
    ]);

    const pruned = pruneSnapshotForTask(snapshot, {
      goal: 'add the product to cart',
      category: PruningTaskCategory.SHOPPING,
    });

    expect(pruned.category).toBe(PruningTaskCategory.SHOPPING);
    expect(pruned.rawElementCount).toBe(4);
    expect(pruned.prunedElementCount).toBe(3);
    expect(pruned.elements.map(element => element.id)).toEqual(expect.arrayContaining([1, 2, 3]));
    expect(pruned.elements.map(element => element.id)).not.toContain(4);
    expect(pruned.promptBlock).toContain('Category: shopping');
    expect(pruned.promptBlock).toContain('Add to Cart');
  });

  it('keeps search inputs and results for search tasks', () => {
    const snapshot = makeSnapshot([
      makeElement({
        id: 10,
        role: 'searchbox',
        text: 'Search',
        importance: 950,
        clickable: true,
      }),
      makeElement({
        id: 11,
        role: 'link',
        text: 'Best Trail Shoes',
        href: '/trail-shoes',
        importance: 900,
        clickable: true,
      }),
      makeElement({
        id: 12,
        role: 'text',
        text: 'Footer links',
        importance: 50,
      }),
    ]);

    const pruned = pruneSnapshotForTask(snapshot, {
      goal: 'search for trail shoes and open the best result',
      category: PruningTaskCategory.SEARCH,
    });

    expect(pruned.elements.map(element => element.id)).toEqual(expect.arrayContaining([10, 11]));
    expect(pruned.elements.map(element => element.id)).not.toContain(12);
  });

  it('keeps Amazon-style searchbox fields in shopping tasks', () => {
    const snapshot = makeSnapshot([
      makeElement({
        id: 30,
        role: 'searchbox',
        text: 'field-keywords',
        name: 'field-keywords',
        ariaLabel: 'Search Amazon',
        importance: 220,
        clickable: true,
      }),
      makeElement({
        id: 31,
        role: 'link',
        text: 'Today Deals',
        href: '/deals',
        importance: 900,
        clickable: true,
      }),
      makeElement({
        id: 32,
        role: 'link',
        text: 'Privacy Policy',
        href: '/privacy',
        importance: 40,
        clickable: true,
      }),
    ]);

    const pruned = pruneSnapshotForTask(snapshot, {
      goal: 'search for noise canceling earbuds then open a product',
      category: PruningTaskCategory.SHOPPING,
    });

    expect(pruned.elements.map(element => element.id)).toContain(30);
  });

  it('keeps form fields and submit controls for form-filling tasks', () => {
    const snapshot = makeSnapshot([
      makeElement({
        id: 20,
        role: 'textbox',
        text: 'Email',
        importance: 900,
        clickable: true,
      }),
      makeElement({
        id: 21,
        role: 'textarea',
        text: 'Message',
        importance: 850,
        clickable: true,
      }),
      makeElement({
        id: 22,
        role: 'button',
        text: 'Submit',
        importance: 950,
        clickable: true,
      }),
      makeElement({
        id: 23,
        role: 'link',
        text: 'Company Blog',
        href: '/blog',
        importance: 100,
        clickable: true,
      }),
    ]);

    const pruned = pruneSnapshotForTask(snapshot, {
      goal: 'fill out the contact form and submit it',
      category: PruningTaskCategory.FORM_FILLING,
    });

    expect(pruned.elements.map(element => element.id)).toEqual(
      expect.arrayContaining([20, 21, 22])
    );
    expect(pruned.elements.map(element => element.id)).not.toContain(23);
  });
});
