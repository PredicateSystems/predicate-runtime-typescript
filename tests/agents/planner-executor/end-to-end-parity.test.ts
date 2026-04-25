import { LLMProvider, type LLMResponse } from '../../../src/llm-provider';
import {
  PlannerExecutorAgent,
  TaskCategory,
  StepStatus,
  type AgentRuntime,
  type Snapshot,
  type SnapshotElement,
} from '../../../src/agents/planner-executor';

class ProviderStub extends LLMProvider {
  public readonly calls: Array<{ mode: 'text' | 'vision'; system: string; user: string }> = [];

  constructor(
    private readonly responses: string[] = [],
    private readonly options: { vision?: boolean } = {}
  ) {
    super();
  }

  get modelName(): string {
    return this.options.vision ? 'vision-provider' : 'provider';
  }

  supportsJsonMode(): boolean {
    return true;
  }

  supportsVision(): boolean {
    return this.options.vision ?? false;
  }

  async generate(systemPrompt: string, userPrompt: string): Promise<LLMResponse> {
    this.calls.push({ mode: 'text', system: systemPrompt, user: userPrompt });
    return {
      content: this.responses.length ? this.responses.shift()! : JSON.stringify({ action: 'DONE' }),
      modelName: this.modelName,
      promptTokens: 10,
      completionTokens: 4,
      totalTokens: 14,
    };
  }

  async generateWithImage(
    systemPrompt: string,
    userPrompt: string,
    _imageBase64: string
  ): Promise<LLMResponse> {
    this.calls.push({ mode: 'vision', system: systemPrompt, user: userPrompt });
    return {
      content: this.responses.length ? this.responses.shift()! : 'CLICK(1)',
      modelName: this.modelName,
      promptTokens: 12,
      completionTokens: 5,
      totalTokens: 17,
    };
  }
}

class RuntimeStub implements AgentRuntime {
  public currentUrl: string;
  public clickCalls: number[] = [];
  public gotoCalls: string[] = [];
  public snapshotLimits: number[] = [];

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

  async snapshot(options?: { limit?: number }): Promise<Snapshot | null> {
    this.snapshotLimits.push(options?.limit ?? -1);
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

function makeSnapshot(
  url: string,
  elements: SnapshotElement[],
  extra: Partial<Snapshot> = {}
): Snapshot {
  return {
    url,
    title: 'Test Page',
    elements,
    screenshot: 'ZmFrZQ==',
    ...extra,
  };
}

describe('PlannerExecutorAgent end-to-end parity', () => {
  it('runs planner hints and heuristics before falling back to the executor', async () => {
    const planner = new ProviderStub([
      JSON.stringify({
        action: 'CLICK',
        goal: 'Add the item to cart',
        intent: 'add_to_cart',
        input: 'Add to Cart',
        heuristic_hints: [
          {
            intent_pattern: 'add_to_cart',
            text_patterns: ['add to cart'],
            role_filter: ['button'],
            priority: 10,
          },
        ],
      }),
      JSON.stringify({ action: 'DONE' }),
    ]);
    const executor = new ProviderStub(['CLICK(999)']);
    const runtime = new RuntimeStub('https://shop.test/product', () =>
      makeSnapshot('https://shop.test/product', [
        { id: 1, role: 'button', text: 'Add to Cart', clickable: true, importance: 100 },
        { id: 2, role: 'link', text: 'Product Details', clickable: true, importance: 80 },
      ])
    );

    const agent = new PlannerExecutorAgent({ planner, executor });
    const result = await agent.runStepwise(runtime, {
      task: 'Add the item to cart',
      category: TaskCategory.TRANSACTION,
    });

    expect(result.success).toBe(true);
    expect(runtime.clickCalls).toEqual([1]);
    expect(executor.calls).toHaveLength(0);
  });

  it('replans after a required failure and applies the replacement step', async () => {
    const planner = new ProviderStub([
      JSON.stringify({
        action: 'CLICK',
        goal: 'Go to checkout',
        intent: 'checkout button',
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
      JSON.stringify({ action: 'DONE' }),
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
  });

  it('recovers from over-pruning before escalating to vision', async () => {
    const planner = new ProviderStub([
      JSON.stringify({
        action: 'CLICK',
        goal: 'Add the product to the wishlist',
        intent: 'add to wishlist button',
        required: true,
      }),
      JSON.stringify({ action: 'DONE' }),
    ]);
    const executor = new ProviderStub(['NONE', 'CLICK(42)'], { vision: true });
    const runtime = new RuntimeStub('https://shop.test/product', () =>
      makeSnapshot('https://shop.test/product', [
        {
          id: 40,
          role: 'link',
          text: 'Trail Shoe',
          href: '/product/trail-shoe',
          importance: 900,
          clickable: true,
          inDominantGroup: true,
        },
        { id: 41, role: 'button', text: 'Add to Cart', clickable: true, importance: 950 },
        { id: 42, role: 'button', text: 'Add to Wishlist', clickable: true, importance: 300 },
        { id: 45, role: 'text', text: '$129.00', importance: 850, nearbyText: 'Price' },
        {
          id: 43,
          role: 'link',
          text: 'Privacy Policy',
          href: '/privacy',
          clickable: true,
          importance: 50,
        },
        {
          id: 44,
          role: 'link',
          text: 'Shipping details',
          href: '/shipping',
          clickable: true,
          importance: 75,
        },
        {
          id: 46,
          role: 'link',
          text: 'Product details',
          href: '/details',
          clickable: true,
          importance: 250,
          inDominantGroup: true,
        },
      ])
    );

    const agent = new PlannerExecutorAgent({
      planner,
      executor,
      intentHeuristics: {
        findElementForIntent: () => null,
        priorityOrder: () => [],
      },
      config: {
        stepwise: { maxSteps: 2 },
        snapshot: {
          limitBase: 50,
          limitMax: 50,
          limitStep: 10,
          pruningMaxRelaxation: 0,
        },
        retry: { verifyTimeoutMs: 20, verifyPollMs: 1, maxReplans: 0 },
      },
    });

    const result = await agent.runStepwise(runtime, { task: 'Add the product to the wishlist' });

    expect(result.success).toBe(true);
    expect(runtime.clickCalls).toEqual([42]);
    expect(executor.calls.map(call => call.mode)).toEqual(['text', 'text']);
  });

  it('stops cleanly at an auth boundary without asking the planner for more work', async () => {
    const planner = new ProviderStub([
      JSON.stringify({ action: 'DONE', reasoning: 'should never be needed' }),
    ]);
    const executor = new ProviderStub();
    const runtime = new RuntimeStub('https://shop.test/login', () =>
      makeSnapshot('https://shop.test/login', [
        { id: 1, role: 'heading', text: 'Sign in', importance: 100 },
      ])
    );

    const agent = new PlannerExecutorAgent({
      planner,
      executor,
      config: {
        authBoundary: {
          enabled: true,
          stopOnAuth: true,
          urlPatterns: ['/login'],
          authSuccessMessage: 'Stopped at authentication boundary',
        },
      },
    });
    const result = await agent.runStepwise(runtime, {
      task: 'Stop when authentication is required',
    });

    expect(result.success).toBe(true);
    expect(result.error).toBe('Stopped at authentication boundary');
    expect(planner.calls).toHaveLength(0);
  });

  it('routes vision-required snapshots through the vision executor path', async () => {
    const planner = new ProviderStub([
      JSON.stringify({
        action: 'CLICK',
        goal: 'Continue through the canvas gate',
        intent: 'continue button',
        verify: [],
      }),
      JSON.stringify({ action: 'DONE' }),
    ]);
    const executor = new ProviderStub(['CLICK(99)'], { vision: true });
    const runtime = new RuntimeStub('https://shop.test/canvas', () =>
      makeSnapshot(
        'https://shop.test/canvas',
        [{ id: 99, role: 'button', text: 'Continue', clickable: true, importance: 100 }],
        { status: 'require_vision' }
      )
    );

    const agent = new PlannerExecutorAgent({
      planner,
      executor,
      config: { stepwise: { maxSteps: 2 } },
    });
    const result = await agent.runStepwise(runtime, { task: 'Continue through the canvas flow' });

    expect(result.success).toBe(true);
    expect(result.fallbackUsed).toBe(true);
    expect(executor.calls.map(call => call.mode)).toEqual(['vision']);
  });
});
