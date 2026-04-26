export enum TaskCategory {
  NAVIGATION = 'navigation',
  SEARCH = 'search',
  FORM_FILL = 'form_fill',
  EXTRACTION = 'extraction',
  TRANSACTION = 'transaction',
  VERIFICATION = 'verification',
}

const CATEGORY_ALIASES: Record<string, TaskCategory> = {
  navigation: TaskCategory.NAVIGATION,
  navigate: TaskCategory.NAVIGATION,
  search: TaskCategory.SEARCH,
  form_fill: TaskCategory.FORM_FILL,
  form: TaskCategory.FORM_FILL,
  'form-fill': TaskCategory.FORM_FILL,
  form_filling: TaskCategory.FORM_FILL,
  extraction: TaskCategory.EXTRACTION,
  extract: TaskCategory.EXTRACTION,
  transaction: TaskCategory.TRANSACTION,
  checkout: TaskCategory.TRANSACTION,
  verification: TaskCategory.VERIFICATION,
  verify: TaskCategory.VERIFICATION,
};

export function normalizeTaskCategory(value?: string | TaskCategory | null): TaskCategory | null {
  if (!value) {
    return null;
  }

  if (Object.values(TaskCategory).includes(value as TaskCategory)) {
    return value as TaskCategory;
  }

  const normalized = value
    .toLowerCase()
    .trim()
    .replace(/[\s-]+/g, '_');
  return CATEGORY_ALIASES[normalized] ?? null;
}
