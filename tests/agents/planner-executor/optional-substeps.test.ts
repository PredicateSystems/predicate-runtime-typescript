import { LLMProvider, type LLMResponse } from '../../../src/llm-provider';
import {
  PlannerExecutorAgent,
  type AgentRuntime,
  type Snapshot,
} from '../../../src/agents/planner-executor';

class ProviderStub extends LLMProvider {
  private responses: string[];
  public calls: Array<{ system: string; user: string; options?: any }> = [];

  constructor(responses: string[] = []) {
    super();
    this.responses = [...responses];
  }

  get modelName(): string {
    return 'stub';
  }

  supportsJsonMode(): boolean {
    return true;
  }

  async generate(
    systemPrompt: string,
    userPrompt: string,
    options: Record<string, any> = {}
  ): Promise<LLMResponse> {
    this.calls.push({ system: systemPrompt, user: userPrompt, options });
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

class RuntimeStub implements AgentRuntime {
  public currentUrl: string;
  public clickCalls: number[] = [];
  public gotoCalls: string[] = [];

  constructor(
    initialUrl: string,
    private readonly snapshotFactory: (runtime: RuntimeStub) => Snapshot | null,
    private readonly handlers: {
      onGoto?: (url: string, runtime: RuntimeStub) => Promise<void> | void;
    } = {}
  ) {
    this.currentUrl = initialUrl;
  }

  async snapshot(): Promise<Snapshot | null> {
    const snap = this.snapshotFactory(this);
    if (snap?.url) {
      this.currentUrl = snap.url;
    }
    return snap;
  }

  async goto(url: string): Promise<void> {
    this.gotoCalls.push(url);
    this.currentUrl = url;
    await this.handlers.onGoto?.(url, this);
  }

  async click(elementId: number): Promise<void> {
    this.clickCalls.push(elementId);
  }

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

function makeSnapshot(url: string, elements: Snapshot['elements']): Snapshot {
  return {
    url,
    title: 'Test Page',
    elements,
  };
}

describe('PlannerExecutorAgent optional substeps', () => {
  it('falls back to optionalSubsteps before replanning', async () => {
    const planner = new ProviderStub([
      JSON.stringify({
        action: 'CLICK',
        goal: 'Add to cart',
        intent: 'add to cart button',
        input: 'Add to Cart',
        required: true,
        optional_substeps: [
          {
            id: 2,
            goal: 'Open the product page first',
            action: 'NAVIGATE',
            target: 'https://shop.test/product/123',
            verify: [{ predicate: 'url_contains', args: ['/product/123'] }],
            required: false,
          },
        ],
      }),
      JSON.stringify({ action: 'DONE', reasoning: 'substep recovered the flow' }),
    ]);
    const executor = new ProviderStub(['NONE']);
    const runtime = new RuntimeStub('https://shop.test/results', rt =>
      makeSnapshot(rt.currentUrl, [{ id: 1, role: 'link', text: 'Product', clickable: true }])
    );

    const agent = new PlannerExecutorAgent({
      planner,
      executor,
      config: { retry: { verifyTimeoutMs: 20, verifyPollMs: 1, maxReplans: 2 } },
    });
    const result = await agent.runStepwise(runtime, { task: 'Add the first result to cart' });

    expect(result.success).toBe(true);
    expect(result.replansUsed).toBe(0);
    expect(runtime.gotoCalls).toEqual(['https://shop.test/product/123']);
    expect(planner.calls).toHaveLength(2);
  });

  it('honors required=false by continuing without replanning after failure', async () => {
    const planner = new ProviderStub([
      JSON.stringify({
        action: 'CLICK',
        goal: 'Dismiss optional promo',
        intent: 'close promo button',
        input: 'No Thanks',
        required: false,
      }),
      JSON.stringify({ action: 'DONE', reasoning: 'optional step can be skipped' }),
    ]);
    const executor = new ProviderStub(['NONE']);
    const runtime = new RuntimeStub('https://shop.test/home', rt =>
      makeSnapshot(rt.currentUrl, [{ id: 1, role: 'button', text: 'No Thanks', clickable: true }])
    );

    const agent = new PlannerExecutorAgent({
      planner,
      executor,
      config: { retry: { verifyTimeoutMs: 20, verifyPollMs: 1, maxReplans: 2 } },
    });
    const result = await agent.runStepwise(runtime, { task: 'Continue shopping' });

    expect(result.success).toBe(true);
    expect(result.replansUsed).toBe(0);
    expect(planner.calls).toHaveLength(2);
    expect(result.stepOutcomes[0].error).toContain('Executor could not find suitable element');
  });
});
