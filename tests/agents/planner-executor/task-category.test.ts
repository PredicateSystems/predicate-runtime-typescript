import {
  TaskCategory,
  normalizeTaskCategory,
} from '../../../src/agents/planner-executor/task-category';

describe('TaskCategory', () => {
  it('defines the Python parity categories', () => {
    expect(TaskCategory.NAVIGATION).toBe('navigation');
    expect(TaskCategory.SEARCH).toBe('search');
    expect(TaskCategory.FORM_FILL).toBe('form_fill');
    expect(TaskCategory.EXTRACTION).toBe('extraction');
    expect(TaskCategory.TRANSACTION).toBe('transaction');
    expect(TaskCategory.VERIFICATION).toBe('verification');
  });

  it('normalizes enum values and lowercase strings', () => {
    expect(normalizeTaskCategory(TaskCategory.SEARCH)).toBe(TaskCategory.SEARCH);
    expect(normalizeTaskCategory('search')).toBe(TaskCategory.SEARCH);
    expect(normalizeTaskCategory('transaction')).toBe(TaskCategory.TRANSACTION);
  });

  it('normalizes common aliases used by planner-executor flows', () => {
    expect(normalizeTaskCategory('form')).toBe(TaskCategory.FORM_FILL);
    expect(normalizeTaskCategory('form-filling')).toBe(TaskCategory.FORM_FILL);
    expect(normalizeTaskCategory('verify')).toBe(TaskCategory.VERIFICATION);
    expect(normalizeTaskCategory('navigate')).toBe(TaskCategory.NAVIGATION);
  });

  it('returns null for unknown categories', () => {
    expect(normalizeTaskCategory('shopping')).toBeNull();
    expect(normalizeTaskCategory(undefined)).toBeNull();
  });
});
