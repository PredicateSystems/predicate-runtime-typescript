import { LLMProvider, type LLMResponse } from '../../../src/llm-provider';
import {
  PlannerExecutorAgent,
  normalizePlan,
  type AgentRuntime,
  type Snapshot,
} from '../../../src/agents/planner-executor';

class PlannerStub extends LLMProvider {
  constructor(private readonly responses: string[]) {
    super();
  }

  get modelName(): string {
    return 'planner-stub';
  }

  supportsJsonMode(): boolean {
    return true;
  }

  async generate(): Promise<LLMResponse> {
    const content = this.responses.length
      ? this.responses.shift()!
      : JSON.stringify({ action: 'DONE' });
    return {
      content,
      modelName: this.modelName,
      promptTokens: 10,
      completionTokens: 5,
      totalTokens: 15,
    };
  }
}

class ExecutorStub extends LLMProvider {
  public calls = 0;

  get modelName(): string {
    return 'executor-stub';
  }

  supportsJsonMode(): boolean {
    return false;
  }

  async generate(): Promise<LLMResponse> {
    this.calls += 1;
    return {
      content: 'NONE',
      modelName: this.modelName,
      promptTokens: 4,
      completionTokens: 1,
      totalTokens: 5,
    };
  }
}

class RuntimeStub implements AgentRuntime {
  public currentUrl: string;
  public gotoCalls: string[] = [];

  constructor(
    initialUrl: string,
    private readonly snapshotValue: Snapshot
  ) {
    this.currentUrl = initialUrl;
  }

  async snapshot(): Promise<Snapshot | null> {
    return { ...this.snapshotValue, url: this.currentUrl };
  }

  async goto(url: string): Promise<void> {
    this.gotoCalls.push(url);
    this.currentUrl = url;
  }

  async click(): Promise<void> {}

  async type(): Promise<void> {}

  async pressKey(): Promise<void> {}

  async scroll(): Promise<void> {}

  async getCurrentUrl(): Promise<string> {
    return this.currentUrl;
  }

  async getViewportHeight(): Promise<number> {
    return 1000;
  }

  async scrollBy(): Promise<boolean> {
    return true;
  }
}

describe('plan normalization', () => {
  it('normalizes Python-style fields, aliases, predicates, and heuristic hints recursively', () => {
    const normalized = normalizePlan({
      task: 'Open the cart',
      steps: [
        {
          id: '1',
          goal: 'Go to cart',
          action: 'go_to',
          url: 'https://shop.test/cart',
          stop_if_true: true,
          heuristic_hints: [
            {
              intent_pattern: 'checkout',
              text_patterns: ['checkout'],
              role_filter: ['button'],
              priority: 9,
            },
          ],
          verify: ["url_contains('/cart')"],
          optional_substeps: [
            {
              id: '2',
              goal: 'Wait for page',
              action: 'wait',
              verify: ['exists(Cart)'],
            },
          ],
        },
      ],
    });

    expect(normalized).toEqual({
      task: 'Open the cart',
      steps: [
        {
          id: 1,
          goal: 'Go to cart',
          action: 'NAVIGATE',
          target: 'https://shop.test/cart',
          stopIfTrue: true,
          heuristicHints: [
            {
              intentPattern: 'checkout',
              textPatterns: ['checkout'],
              roleFilter: ['button'],
              priority: 9,
              attributePatterns: {},
            },
          ],
          verify: [{ predicate: 'url_contains', args: ['/cart'] }],
          optionalSubsteps: [
            {
              id: 2,
              goal: 'Wait for page',
              action: 'WAIT',
              verify: [{ predicate: 'exists', args: ['Cart'] }],
            },
          ],
        },
      ],
    });
  });

  it('normalizes stepwise planner responses before execution', async () => {
    const planner = new PlannerStub([
      JSON.stringify({
        action: 'go_to',
        url: 'https://shop.test/results',
        verify: ["url_contains('/results')"],
        heuristic_hints: [
          {
            intent_pattern: 'open_results',
            text_patterns: ['results'],
            role_filter: ['link'],
            priority: 5,
          },
        ],
      }),
      JSON.stringify({ action: 'done' }),
    ]);
    const executor = new ExecutorStub();
    const runtime = new RuntimeStub('https://shop.test', {
      url: 'https://shop.test',
      title: 'Start',
      elements: [{ id: 1, role: 'link', text: 'Results', clickable: true }],
    });

    const agent = new PlannerExecutorAgent({ planner, executor });
    const result = await agent.runStepwise(runtime, { task: 'Open the results page' });

    expect(result.success).toBe(true);
    expect(runtime.gotoCalls).toEqual(['https://shop.test/results']);
    expect(executor.calls).toBe(0);
  });
});
