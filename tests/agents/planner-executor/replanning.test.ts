import { LLMProvider, type LLMResponse } from '../../../src/llm-provider';
import {
  PlannerExecutorAgent,
  StepStatus,
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
      onClick?: (elementId: number, runtime: RuntimeStub) => Promise<void> | void;
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
    await this.handlers.onClick?.(elementId, this);
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

describe('PlannerExecutorAgent replanning', () => {
  it('triggers a replan after a required step fails and applies replacement steps', async () => {
    const planner = new ProviderStub([
      JSON.stringify({
        action: 'CLICK',
        intent: 'checkout button',
        input: 'Checkout',
        verify: [{ predicate: 'url_contains', args: ['/checkout'] }],
        required: true,
      }),
      JSON.stringify({
        mode: 'patch',
        replace_steps: [
          {
            id: 1,
            step: {
              id: 1,
              goal: 'Navigate directly to checkout',
              action: 'NAVIGATE',
              target: 'https://shop.test/checkout',
              verify: [{ predicate: 'url_contains', args: ['/checkout'] }],
              required: true,
            },
          },
        ],
      }),
      JSON.stringify({ action: 'DONE', reasoning: 'done after repair' }),
    ]);
    const executor = new ProviderStub(['NONE']);
    const runtime = new RuntimeStub('https://shop.test/cart', rt =>
      makeSnapshot(rt.currentUrl, [
        { id: 1, role: 'button', text: 'Checkout', clickable: true, importance: 100 },
      ])
    );

    const agent = new PlannerExecutorAgent({
      planner,
      executor,
      config: { retry: { verifyTimeoutMs: 20, verifyPollMs: 1, maxReplans: 1 } },
    });
    const result = await agent.runStepwise(runtime, { task: 'Finish checkout' });

    expect(result.success).toBe(true);
    expect(result.replansUsed).toBe(1);
    expect(result.stepOutcomes[0].status).toBe(StepStatus.FAILED);
    expect(runtime.gotoCalls).toEqual(['https://shop.test/checkout']);
    expect(planner.calls).toHaveLength(3);
    expect(planner.calls[1].system).toContain('JSON patch');
  });

  it('handles STUCK by requesting a repair step instead of terminating immediately', async () => {
    const planner = new ProviderStub([
      JSON.stringify({
        action: 'STUCK',
        reasoning: 'search results are hidden behind a modal',
      }),
      JSON.stringify({
        mode: 'patch',
        replace_steps: [
          {
            id: 1,
            step: {
              id: 1,
              goal: 'Open the product details page directly',
              action: 'NAVIGATE',
              target: 'https://shop.test/product/123',
              verify: [{ predicate: 'url_contains', args: ['/product/123'] }],
              required: true,
            },
          },
        ],
      }),
      JSON.stringify({ action: 'DONE', reasoning: 'recovered' }),
    ]);
    const executor = new ProviderStub();
    const runtime = new RuntimeStub('https://shop.test/results', rt =>
      makeSnapshot(rt.currentUrl, [{ id: 1, role: 'link', text: 'Product', clickable: true }])
    );

    const agent = new PlannerExecutorAgent({
      planner,
      executor,
      config: { retry: { verifyTimeoutMs: 20, verifyPollMs: 1, maxReplans: 1 } },
    });
    const result = await agent.runStepwise(runtime, { task: 'Open the first product' });

    expect(result.success).toBe(true);
    expect(result.replansUsed).toBe(1);
    expect(result.stepOutcomes[0].actionTaken).toBe('STUCK');
    expect(runtime.gotoCalls).toEqual(['https://shop.test/product/123']);
  });

  it('replans STUCK even when a recovery checkpoint exists', async () => {
    const planner = new ProviderStub([
      JSON.stringify({
        action: 'NAVIGATE',
        target: 'https://shop.test/results',
        verify: [{ predicate: 'url_contains', args: ['/results'] }],
      }),
      JSON.stringify({
        action: 'STUCK',
        reasoning: 'results page is blocked',
      }),
      JSON.stringify({
        mode: 'patch',
        replace_steps: [
          {
            id: 2,
            step: {
              id: 2,
              goal: 'Navigate directly to a safe recovery page',
              action: 'NAVIGATE',
              target: 'https://shop.test/recovered',
              verify: [{ predicate: 'url_contains', args: ['/recovered'] }],
              required: true,
            },
          },
        ],
      }),
      JSON.stringify({ action: 'DONE', reasoning: 'recovered after stuck' }),
    ]);
    const executor = new ProviderStub();
    const runtime = new RuntimeStub('https://shop.test/home', rt =>
      makeSnapshot(rt.currentUrl, [{ id: 1, role: 'link', text: 'Result', clickable: true }])
    );

    const agent = new PlannerExecutorAgent({
      planner,
      executor,
      config: { retry: { verifyTimeoutMs: 20, verifyPollMs: 1, maxReplans: 1 } },
    });
    const result = await agent.runStepwise(runtime, { task: 'Open a result despite blockers' });

    expect(result.success).toBe(true);
    expect(result.replansUsed).toBe(1);
    expect(planner.calls[2].system).toContain('JSON patch');
    expect(runtime.gotoCalls).toEqual(['https://shop.test/results', 'https://shop.test/recovered']);
  });

  it('accepts string replace_steps ids from repair patches', async () => {
    const planner = new ProviderStub([
      JSON.stringify({
        action: 'CLICK',
        intent: 'checkout button',
        input: 'Checkout',
        verify: [{ predicate: 'url_contains', args: ['/checkout'] }],
        required: true,
      }),
      JSON.stringify({
        mode: 'patch',
        replace_steps: [
          {
            id: '1',
            step: {
              id: '1',
              goal: 'Navigate directly to checkout',
              action: 'NAVIGATE',
              target: 'https://shop.test/checkout',
              verify: [{ predicate: 'url_contains', args: ['/checkout'] }],
              required: true,
            },
          },
        ],
      }),
      JSON.stringify({ action: 'DONE', reasoning: 'repaired with string ids' }),
    ]);
    const executor = new ProviderStub(['NONE']);
    const runtime = new RuntimeStub('https://shop.test/cart', rt =>
      makeSnapshot(rt.currentUrl, [
        { id: 1, role: 'button', text: 'Checkout', clickable: true, importance: 100 },
      ])
    );

    const agent = new PlannerExecutorAgent({
      planner,
      executor,
      config: { retry: { verifyTimeoutMs: 20, verifyPollMs: 1, maxReplans: 1 } },
    });
    const result = await agent.runStepwise(runtime, { task: 'Finish checkout with repair patch' });

    expect(result.success).toBe(true);
    expect(result.replansUsed).toBe(1);
    expect(runtime.gotoCalls).toEqual(['https://shop.test/checkout']);
  });

  it('stops replanning at maxReplans and includes prior history in repair prompts', async () => {
    const planner = new ProviderStub([
      JSON.stringify({
        action: 'CLICK',
        intent: 'place order button',
        input: 'Place Order',
        verify: [{ predicate: 'url_contains', args: ['/done'] }],
        required: true,
      }),
      JSON.stringify({
        mode: 'patch',
        replace_steps: [
          {
            id: 1,
            step: {
              id: 1,
              goal: 'Try the alternate place order control',
              action: 'CLICK',
              intent: 'alternate place order button',
              input: 'Place Order',
              verify: [{ predicate: 'url_contains', args: ['/done'] }],
              required: true,
            },
          },
        ],
      }),
    ]);
    const executor = new ProviderStub(['NONE', 'NONE']);
    const runtime = new RuntimeStub('https://shop.test/checkout', rt =>
      makeSnapshot(rt.currentUrl, [
        { id: 1, role: 'button', text: 'Place Order', clickable: true, importance: 100 },
      ])
    );

    const agent = new PlannerExecutorAgent({
      planner,
      executor,
      config: { retry: { verifyTimeoutMs: 20, verifyPollMs: 1, maxReplans: 1 } },
    });
    const result = await agent.runStepwise(runtime, { task: 'Submit the order' });

    expect(result.success).toBe(false);
    expect(result.replansUsed).toBe(1);
    expect(planner.calls).toHaveLength(2);
    expect(planner.calls[1].user).toContain('Failure classification: element-not-found');
    expect(planner.calls[1].user).toContain('1. CLICK(place order button) -> failed');
    expect(result.error).toContain('max replans');
  });
});
