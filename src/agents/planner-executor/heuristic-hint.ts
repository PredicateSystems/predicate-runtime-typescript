import type { SnapshotElement } from './plan-models';

export interface HeuristicHintInput {
  intentPattern?: string;
  intent_pattern?: string;
  textPatterns?: string[];
  text_patterns?: string[];
  roleFilter?: string[];
  role_filter?: string[];
  priority?: number;
  attributePatterns?: Record<string, string>;
  attribute_patterns?: Record<string, string>;
}

type ElementWithAttributes = SnapshotElement & {
  attributes?: Record<string, string>;
};

export class HeuristicHint {
  intentPattern: string;
  textPatterns: string[];
  roleFilter: string[];
  priority: number;
  attributePatterns: Record<string, string>;

  constructor(input: HeuristicHintInput) {
    this.intentPattern = input.intentPattern ?? input.intent_pattern ?? '';
    this.textPatterns = input.textPatterns ?? input.text_patterns ?? [];
    this.roleFilter = input.roleFilter ?? input.role_filter ?? [];
    this.priority = input.priority ?? 0;
    this.attributePatterns = input.attributePatterns ?? input.attribute_patterns ?? {};
  }

  matchesIntent(intent: string): boolean {
    if (!intent || !this.intentPattern) {
      return false;
    }

    return intent.toLowerCase().includes(this.intentPattern.toLowerCase());
  }

  matchesElement(element: SnapshotElement): boolean {
    const role = (element.role ?? '').toLowerCase();
    if (
      this.roleFilter.length > 0 &&
      !this.roleFilter.some(candidate => candidate.toLowerCase() === role)
    ) {
      return false;
    }

    const textCandidates = [element.text, element.ariaLabel, element.name]
      .filter((value): value is string => typeof value === 'string')
      .map(value => value.toLowerCase());
    if (
      this.textPatterns.length > 0 &&
      !this.textPatterns.some(pattern =>
        textCandidates.some(candidate => candidate.includes(pattern.toLowerCase()))
      )
    ) {
      return false;
    }

    return this.matchesAttributes(element as ElementWithAttributes);
  }

  private matchesAttributes(element: ElementWithAttributes): boolean {
    const entries = Object.entries(this.attributePatterns);
    if (entries.length === 0) {
      return true;
    }

    for (const [attributeName, attributePattern] of entries) {
      const candidate = this.readAttribute(element, attributeName);
      if (!candidate.toLowerCase().includes(attributePattern.toLowerCase())) {
        return false;
      }
    }

    return true;
  }

  private readAttribute(element: ElementWithAttributes, attributeName: string): string {
    const directValue = (element as unknown as Record<string, unknown>)[attributeName];
    if (typeof directValue === 'string') {
      return directValue;
    }

    const attributes = element.attributes ?? {};
    return attributes[attributeName] ?? '';
  }
}
