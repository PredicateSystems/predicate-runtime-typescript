import type { SnapshotElement } from './plan-models';
import type { IntentHeuristics } from './planner-executor-agent';
import { COMMON_HINTS } from './common-hints';
import { HeuristicHint, type HeuristicHintInput } from './heuristic-hint';
import { TaskCategory } from './task-category';
import type { LearnedTargetFingerprint } from './profile-types';

export interface ComposableHeuristicsOptions {
  staticHeuristics?: IntentHeuristics;
  taskCategory?: TaskCategory | null;
  useCommonHints?: boolean;
  /** Learned fingerprints from previous successful runs */
  learnedFingerprints?: LearnedTargetFingerprint[];
}

/** Minimum confidence for a learned fingerprint to be used as a hint */
const MIN_FINGERPRINT_CONFIDENCE = 0.3;

export class ComposableHeuristics implements IntentHeuristics {
  private readonly staticHeuristics: IntentHeuristics | null;
  private readonly taskCategory: TaskCategory | null;
  private readonly useCommonHints: boolean;
  private currentHints: HeuristicHint[] = [];
  private readonly learnedFingerprints: LearnedTargetFingerprint[];

  constructor(options: ComposableHeuristicsOptions = {}) {
    this.staticHeuristics = options.staticHeuristics ?? null;
    this.taskCategory = options.taskCategory ?? null;
    this.useCommonHints = options.useCommonHints ?? true;
    this.learnedFingerprints = (options.learnedFingerprints ?? []).filter(
      fp => fp.confidence >= MIN_FINGERPRINT_CONFIDENCE
    );
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

    // 1. Check current step hints (highest priority)
    for (const hint of this.currentHints) {
      if (hint.matchesIntent(intent)) {
        const elementId = this.matchHint(hint, elements);
        if (elementId !== null) {
          return elementId;
        }
      }
    }

    // 2. Check learned fingerprints (dynamic hints from successful past runs)
    const learnedMatch = this.matchLearnedFingerprint(intent, elements);
    if (learnedMatch !== null) {
      return learnedMatch;
    }

    // 3. Check common hints (built-in)
    if (this.useCommonHints) {
      const commonHint = this.getCommonHintForIntent(intent);
      if (commonHint) {
        const elementId = this.matchHint(commonHint, elements);
        if (elementId !== null) {
          return elementId;
        }
      }
    }

    // 4. Check static heuristics
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

    // 5. Fall back to task category defaults
    return this.matchTaskCategoryDefaults(elements);
  }

  priorityOrder(): string[] {
    const patterns = [
      ...this.currentHints.map(hint => hint.intentPattern),
      ...this.learnedFingerprints.map(fp => fp.intent),
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

  /**
   * Match learned fingerprints against current snapshot elements.
   * Fingerprints are sorted by confidence (descending) so the most
   * reliable past success is tried first.
   */
  private matchLearnedFingerprint(intent: string, elements: SnapshotElement[]): number | null {
    if (this.learnedFingerprints.length === 0) {
      return null;
    }

    const normalizedIntent = intent.toLowerCase().replace(/[\s-]+/g, '_');
    const sorted = [...this.learnedFingerprints].sort((a, b) => b.confidence - a.confidence);

    for (const fp of sorted) {
      // Match intent: exact or substring match
      const fpIntent = fp.intent.toLowerCase().replace(/[\s-]+/g, '_');
      if (
        fpIntent !== normalizedIntent &&
        !normalizedIntent.includes(fpIntent) &&
        !fpIntent.includes(normalizedIntent)
      ) {
        continue;
      }

      for (const element of elements) {
        if (this.fingerprintMatchesElement(fp, element)) {
          return element.id;
        }
      }
    }

    return null;
  }

  /**
   * Check if a learned fingerprint matches a snapshot element.
   * Uses token overlap scoring with a minimum threshold.
   */
  private fingerprintMatchesElement(
    fp: LearnedTargetFingerprint,
    element: SnapshotElement
  ): boolean {
    let score = 0;
    let maxScore = 0;

    // Role match (weight: 2)
    if (fp.role) {
      maxScore += 2;
      if ((element.role ?? '').toLowerCase() === fp.role) {
        score += 2;
      }
    }

    // Text token overlap (weight: up to 3)
    if (fp.textTokens && fp.textTokens.length > 0) {
      maxScore += 3;
      const elementText = [element.text, element.ariaLabel, element.name]
        .filter((v): v is string => typeof v === 'string')
        .join(' ')
        .toLowerCase();
      const elementTokens = elementText.split(/\s+/).filter(t => t.length > 0);
      const matchingTokens = fp.textTokens.filter(ft =>
        elementTokens.some(et => et === ft || et.includes(ft))
      );
      if (matchingTokens.length > 0) {
        score += Math.min(3, Math.ceil((matchingTokens.length / fp.textTokens.length) * 3));
      }
    }

    // ARIA token overlap (weight: up to 2)
    if (fp.ariaTokens && fp.ariaTokens.length > 0) {
      maxScore += 2;
      const ariaText = [element.ariaLabel, element.name]
        .filter((v): v is string => typeof v === 'string')
        .join(' ')
        .toLowerCase();
      const ariaTokens = ariaText.split(/\s+/).filter(t => t.length > 0);
      const matchingTokens = fp.ariaTokens.filter(at =>
        ariaTokens.some(et => et === at || et.includes(at))
      );
      if (matchingTokens.length > 0) {
        score += Math.min(2, Math.ceil((matchingTokens.length / fp.ariaTokens.length) * 2));
      }
    }

    // href path pattern match (weight: 2)
    if (fp.hrefPathPattern) {
      maxScore += 2;
      const href = element.href || '';
      if (href.toLowerCase().includes(fp.hrefPathPattern)) {
        score += 2;
      }
    }

    // Require at least 50% of available score to consider it a match
    if (maxScore === 0) return false;
    return score / maxScore >= 0.5;
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
        return this.matchKeywordElements(
          elements,
          ['button', 'textbox', 'searchbox', 'combobox'],
          ['search', 'find', 'go']
        );
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
