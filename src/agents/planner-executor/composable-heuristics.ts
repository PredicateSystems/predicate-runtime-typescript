import type { SnapshotElement } from './plan-models';
import type { IntentHeuristics } from './planner-executor-agent';
import { COMMON_HINTS } from './common-hints';
import { HeuristicHint, type HeuristicHintInput } from './heuristic-hint';
import { TaskCategory } from './task-category';

export interface ComposableHeuristicsOptions {
  staticHeuristics?: IntentHeuristics;
  taskCategory?: TaskCategory | null;
  useCommonHints?: boolean;
}

export class ComposableHeuristics implements IntentHeuristics {
  private readonly staticHeuristics: IntentHeuristics | null;
  private readonly taskCategory: TaskCategory | null;
  private readonly useCommonHints: boolean;
  private currentHints: HeuristicHint[] = [];

  constructor(options: ComposableHeuristicsOptions = {}) {
    this.staticHeuristics = options.staticHeuristics ?? null;
    this.taskCategory = options.taskCategory ?? null;
    this.useCommonHints = options.useCommonHints ?? true;
  }

  setStepHints(hints?: Array<HeuristicHint | HeuristicHintInput> | null): void {
    if (!hints || hints.length === 0) {
      this.currentHints = [];
      return;
    }

    this.currentHints = hints
      .map(hint => (hint instanceof HeuristicHint ? hint : new HeuristicHint(hint)))
      .filter(hint => hint.intentPattern.length > 0)
      .sort((left, right) => right.priority - left.priority);
  }

  clearStepHints(): void {
    this.currentHints = [];
  }

  findElementForIntent(
    intent: string,
    elements: SnapshotElement[],
    url: string,
    goal: string
  ): number | null {
    if (!intent || elements.length === 0) {
      return null;
    }

    for (const hint of this.currentHints) {
      if (hint.matchesIntent(intent)) {
        const elementId = this.matchHint(hint, elements);
        if (elementId !== null) {
          return elementId;
        }
      }
    }

    if (this.useCommonHints) {
      const commonHint = this.getCommonHintForIntent(intent);
      if (commonHint) {
        const elementId = this.matchHint(commonHint, elements);
        if (elementId !== null) {
          return elementId;
        }
      }
    }

    if (this.staticHeuristics) {
      try {
        const elementId = this.staticHeuristics.findElementForIntent(intent, elements, url, goal);
        if (elementId !== null) {
          return elementId;
        }
      } catch {
        // Static heuristics are optional and should not break agent execution.
      }
    }

    return this.matchTaskCategoryDefaults(elements);
  }

  priorityOrder(): string[] {
    const patterns = [
      ...this.currentHints.map(hint => hint.intentPattern),
      ...(this.useCommonHints ? Object.keys(COMMON_HINTS) : []),
    ];

    if (this.staticHeuristics) {
      try {
        patterns.push(...this.staticHeuristics.priorityOrder());
      } catch {
        // Ignore priority-order failures from optional heuristics.
      }
    }

    return patterns.filter((pattern, index) => patterns.indexOf(pattern) === index);
  }

  private matchHint(hint: HeuristicHint, elements: SnapshotElement[]): number | null {
    for (const element of elements) {
      if (hint.matchesElement(element)) {
        return element.id;
      }
    }

    return null;
  }

  private getCommonHintForIntent(intent: string): HeuristicHint | null {
    const normalized = intent.toLowerCase().replace(/[\s-]+/g, '_');
    if (normalized in COMMON_HINTS) {
      return COMMON_HINTS[normalized as keyof typeof COMMON_HINTS];
    }

    for (const [key, hint] of Object.entries(COMMON_HINTS)) {
      if (normalized.includes(key) || key.includes(normalized)) {
        return hint;
      }
    }

    return null;
  }

  private matchTaskCategoryDefaults(elements: SnapshotElement[]): number | null {
    switch (this.taskCategory) {
      case TaskCategory.TRANSACTION:
        return this.matchKeywordElements(
          elements,
          ['button', 'link'],
          [
            'add to cart',
            'add to bag',
            'buy now',
            'checkout',
            'proceed',
            'submit',
            'confirm',
            'place order',
          ]
        );
      case TaskCategory.FORM_FILL:
        return this.matchKeywordElements(
          elements,
          ['button'],
          ['submit', 'next', 'continue', 'save', 'update']
        );
      case TaskCategory.SEARCH:
        return this.matchKeywordElements(elements, ['button', 'textbox'], ['search', 'find', 'go']);
      case TaskCategory.NAVIGATION:
        return this.matchKeywordElements(elements, ['link'], []);
      default:
        return null;
    }
  }

  private matchKeywordElements(
    elements: SnapshotElement[],
    roles: string[],
    keywords: string[]
  ): number | null {
    for (const element of elements) {
      const role = (element.role ?? '').toLowerCase();
      if (!roles.includes(role)) {
        continue;
      }

      const candidates = [element.text, element.ariaLabel, element.name]
        .filter((value): value is string => typeof value === 'string')
        .map(value => value.toLowerCase());

      if (
        keywords.length === 0 ||
        keywords.some(keyword => candidates.some(candidate => candidate.includes(keyword)))
      ) {
        return element.id;
      }
    }

    return null;
  }
}
