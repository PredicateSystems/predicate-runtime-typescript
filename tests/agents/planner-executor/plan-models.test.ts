import {
  ActionType,
  PlanSchema,
  PlanStepSchema,
  TaskCategory,
  AutomationTaskSchema,
} from '../../../src/agents/planner-executor';

describe('plan models', () => {
  it('supports the richer Python parity action set', () => {
    expect(ActionType.options).toEqual(
      expect.arrayContaining(['NAVIGATE', 'WAIT', 'EXTRACT', 'STUCK', 'DONE'])
    );
  });

  it('parses steps with verification flags, substeps, and heuristic hints', () => {
    const step = PlanStepSchema.parse({
      id: 1,
      goal: 'Add product to cart',
      action: 'CLICK',
      intent: 'add_to_cart',
      verify: [{ predicate: 'url_contains', args: ['/cart'] }],
      required: true,
      stopIfTrue: false,
      optionalSubsteps: [
        {
          id: 2,
          goal: 'Open product page first',
          action: 'NAVIGATE',
          target: 'https://shop.test/product',
          verify: [],
          required: false,
          stopIfTrue: false,
          optionalSubsteps: [],
          heuristicHints: [],
        },
      ],
      heuristicHints: [
        {
          intentPattern: 'add_to_cart',
          textPatterns: ['add to cart', 'buy now'],
          roleFilter: ['button'],
          priority: 10,
        },
      ],
    });

    expect(step.optionalSubsteps).toHaveLength(1);
    expect(step.heuristicHints[0]).toMatchObject({
      intentPattern: 'add_to_cart',
      textPatterns: ['add to cart', 'buy now'],
    });
  });

  it('parses plans that include WAIT, EXTRACT, and STUCK steps', () => {
    const plan = PlanSchema.parse({
      task: 'Wait for results and extract them',
      notes: ['planner note'],
      steps: [
        {
          id: 1,
          goal: 'Wait for results',
          action: 'WAIT',
          verify: [{ predicate: 'exists', args: ['Search Results'] }],
          required: true,
          stopIfTrue: false,
          optionalSubsteps: [],
          heuristicHints: [],
        },
        {
          id: 2,
          goal: 'Extract the results',
          action: 'EXTRACT',
          verify: [],
          required: true,
          stopIfTrue: false,
          optionalSubsteps: [],
          heuristicHints: [],
        },
        {
          id: 3,
          goal: 'Escalate if blocked',
          action: 'STUCK',
          verify: [],
          required: false,
          stopIfTrue: false,
          optionalSubsteps: [],
          heuristicHints: [],
        },
      ],
    });

    expect(plan.steps.map(step => step.action)).toEqual(['WAIT', 'EXTRACT', 'STUCK']);
  });

  it('parses automation task metadata for category-driven execution', () => {
    const task = AutomationTaskSchema.parse({
      task: 'Add the first result to cart',
      startUrl: 'https://shop.test',
      category: TaskCategory.TRANSACTION,
      domainHints: ['ecommerce', 'shop'],
    });

    expect(task.category).toBe(TaskCategory.TRANSACTION);
    expect(task.domainHints).toEqual(['ecommerce', 'shop']);
  });
});
