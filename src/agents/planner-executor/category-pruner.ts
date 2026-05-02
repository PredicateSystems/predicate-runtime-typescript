import type { Snapshot, SnapshotElement } from './plan-models';
import { formatPrunedContext, selectContextElements } from './plan-utils';
import {
  PruningTaskCategory,
  type PruneSnapshotOptions,
  type PrunedSnapshotContext,
} from './pruning-types';
import { TaskCategory } from './task-category';

function textOf(element: SnapshotElement): string {
  return String(element.text || element.name || '').toLowerCase();
}

function hrefOf(element: SnapshotElement): string {
  return String(element.href || '').toLowerCase();
}

function nearbyTextOf(element: SnapshotElement): string {
  return String(element.nearbyText || '').toLowerCase();
}

function combinedTextOf(element: SnapshotElement): string {
  return [element.text, element.name, element.ariaLabel, element.nearbyText, element.href]
    .filter((value): value is string => Boolean(value))
    .join(' ')
    .toLowerCase();
}

function hasSearchCue(element: SnapshotElement): boolean {
  const combined = combinedTextOf(element);
  return (
    combined.includes('search') ||
    combined.includes('keyword') ||
    combined.includes('find') ||
    combined.includes('query')
  );
}

function roleOf(element: SnapshotElement): string {
  return String(element.role || '').toLowerCase();
}

function isInteractive(element: SnapshotElement): boolean {
  return new Set([
    'button',
    'link',
    'textbox',
    'searchbox',
    'combobox',
    'checkbox',
    'radio',
    'slider',
    'tab',
    'menuitem',
    'option',
    'switch',
    'cell',
    'input',
    'select',
    'textarea',
  ]).has(roleOf(element));
}

function allowShopping(element: SnapshotElement): boolean {
  const text = textOf(element);
  const role = roleOf(element);
  const href = hrefOf(element);
  const nearbyText = nearbyTextOf(element);

  if (
    ['button', 'link', 'textbox', 'searchbox', 'combobox'].includes(role) &&
    ['add to cart', 'add to bag', 'buy now', 'checkout', 'cart'].some(token => text.includes(token))
  ) {
    return true;
  }
  if (role === 'link' && href && element.inDominantGroup) {
    return true;
  }
  if (text.includes('$') || nearbyText.includes('price')) {
    return true;
  }
  if (role === 'searchbox') {
    return true;
  }
  if (['textbox', 'combobox', 'input', 'textarea'].includes(role) && hasSearchCue(element)) {
    return true;
  }
  return Boolean(element.inDominantGroup && text.trim().length >= 3);
}

function allowShoppingRelaxed(element: SnapshotElement): boolean {
  if (allowShopping(element)) {
    return true;
  }
  const role = roleOf(element);
  if (['button', 'link'].includes(role) && textOf(element).trim().length >= 2) {
    return true;
  }
  return ['select', 'combobox'].includes(role);
}

function allowShoppingLoose(element: SnapshotElement): boolean {
  return allowShoppingRelaxed(element) || isInteractive(element);
}

function allowFormFilling(element: SnapshotElement): boolean {
  const role = roleOf(element);
  const text = textOf(element);
  if (['textbox', 'searchbox', 'combobox', 'checkbox', 'radio', 'textarea'].includes(role)) {
    return true;
  }
  return (
    role === 'button' &&
    ['submit', 'send', 'continue', 'sign up'].some(token => text.includes(token))
  );
}

function allowFormFillingRelaxed(element: SnapshotElement): boolean {
  if (allowFormFilling(element)) {
    return true;
  }
  return ['button', 'select', 'option'].includes(roleOf(element));
}

function allowSearch(element: SnapshotElement): boolean {
  const role = roleOf(element);
  if (['searchbox', 'textbox', 'combobox'].includes(role)) {
    return true;
  }
  if (role === 'button' && textOf(element).includes('search')) {
    return true;
  }
  return role === 'link' && Boolean(element.href);
}

function allowSearchRelaxed(element: SnapshotElement): boolean {
  if (allowSearch(element)) {
    return true;
  }
  return ['button', 'tab', 'menuitem'].includes(roleOf(element));
}

function allowGeneric(element: SnapshotElement): boolean {
  return ['button', 'link', 'textbox', 'searchbox', 'combobox', 'checkbox', 'radio'].includes(
    roleOf(element)
  );
}

function allowGenericRelaxed(element: SnapshotElement): boolean {
  return isInteractive(element);
}

function blockCommon(element: SnapshotElement): boolean {
  const text = textOf(element);
  const href = hrefOf(element);
  return (
    ['privacy policy', 'terms', 'cookie policy'].some(token => text.includes(token)) ||
    ['/privacy', '/terms', '/cookies'].some(token => href.includes(token))
  );
}

function scoreElement(element: SnapshotElement, goal: string): number {
  const goalTerms = goal
    .toLowerCase()
    .split(/\s+/)
    .filter(term => term.length > 2);
  const text = textOf(element);

  let score = Number(element.importance || 0);
  if (element.clickable) {
    score += 15;
  }
  if (element.inDominantGroup) {
    score += 20;
  }
  if (goalTerms.some(term => text.includes(term))) {
    score += 10;
  }
  if (text.includes('$')) {
    score += 8;
  }
  return score;
}

function getPolicy(
  category: PruningTaskCategory,
  relaxationLevel: number
): {
  maxNodes: number;
  allow: (element: SnapshotElement) => boolean;
  block: (element: SnapshotElement) => boolean;
} {
  if (relaxationLevel >= 3) {
    return {
      maxNodes: 80,
      allow: allowGenericRelaxed,
      block: () => false,
    };
  }

  if (category === PruningTaskCategory.SHOPPING || category === PruningTaskCategory.CHECKOUT) {
    if (relaxationLevel === 0) {
      return { maxNodes: 25, allow: allowShopping, block: blockCommon };
    }
    if (relaxationLevel === 1) {
      return { maxNodes: 40, allow: allowShoppingRelaxed, block: blockCommon };
    }
    return { maxNodes: 60, allow: allowShoppingLoose, block: () => false };
  }

  if (category === PruningTaskCategory.FORM_FILLING) {
    return relaxationLevel === 0
      ? { maxNodes: 20, allow: allowFormFilling, block: blockCommon }
      : { maxNodes: 35 + relaxationLevel * 10, allow: allowFormFillingRelaxed, block: () => false };
  }

  if (category === PruningTaskCategory.SEARCH) {
    return relaxationLevel === 0
      ? { maxNodes: 20, allow: allowSearch, block: blockCommon }
      : { maxNodes: 35 + relaxationLevel * 10, allow: allowSearchRelaxed, block: () => false };
  }

  return relaxationLevel === 0
    ? { maxNodes: 20, allow: allowGeneric, block: blockCommon }
    : { maxNodes: 40 + relaxationLevel * 15, allow: allowGenericRelaxed, block: () => false };
}

export function detectPruningCategory(
  taskCategory: TaskCategory | null,
  goal: string
): PruningTaskCategory | null {
  const normalizedGoal = goal.toLowerCase();

  if (taskCategory === TaskCategory.SEARCH) {
    return PruningTaskCategory.SEARCH;
  }
  if (taskCategory === TaskCategory.FORM_FILL) {
    return PruningTaskCategory.FORM_FILLING;
  }
  if (taskCategory === TaskCategory.TRANSACTION) {
    if (normalizedGoal.includes('checkout')) {
      return PruningTaskCategory.CHECKOUT;
    }
    return PruningTaskCategory.SHOPPING;
  }
  if (normalizedGoal.includes('search') || normalizedGoal.includes('find result')) {
    return PruningTaskCategory.SEARCH;
  }
  if (
    normalizedGoal.includes('form') ||
    normalizedGoal.includes('fill out') ||
    normalizedGoal.includes('submit')
  ) {
    return PruningTaskCategory.FORM_FILLING;
  }
  if (
    ['cart', 'checkout', 'buy', 'product', 'shop', 'wishlist'].some(token =>
      normalizedGoal.includes(token)
    )
  ) {
    return normalizedGoal.includes('checkout')
      ? PruningTaskCategory.CHECKOUT
      : PruningTaskCategory.SHOPPING;
  }
  return null;
}

export function pruneSnapshotForTask(
  snapshot: Snapshot,
  options: PruneSnapshotOptions
): PrunedSnapshotContext {
  const relaxationLevel = Math.max(0, options.relaxationLevel || 0);
  const policy = getPolicy(options.category, relaxationLevel);
  const filtered = (snapshot.elements || []).filter(element => {
    if (policy.block(element)) {
      return false;
    }
    return policy.allow(element);
  });
  const elements = filtered
    .sort((left, right) => scoreElement(right, options.goal) - scoreElement(left, options.goal))
    .slice(0, policy.maxNodes);
  const actionableElementCount = selectContextElements(elements, elements.length || 1).length;

  return {
    category: options.category,
    snapshot,
    elements,
    promptBlock: formatPrunedContext({
      category: options.category,
      elements,
      relaxationLevel,
      rawElementCount: snapshot.elements.length,
      prunedElementCount: elements.length,
      actionableElementCount,
    }),
    relaxationLevel,
    rawElementCount: snapshot.elements.length,
    prunedElementCount: elements.length,
    actionableElementCount,
  };
}

export { PruningTaskCategory } from './pruning-types';
