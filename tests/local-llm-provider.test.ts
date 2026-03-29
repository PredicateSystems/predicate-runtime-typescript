import { LocalLLMProvider, LocalVisionLLMProvider, OllamaProvider } from '../src/llm-provider';

describe('LocalLLMProvider (OpenAI-compatible)', () => {
  const originalFetch = (globalThis as any).fetch;

  afterEach(() => {
    (globalThis as any).fetch = originalFetch;
  });

  it('should call /chat/completions and parse response', async () => {
    (globalThis as any).fetch = jest.fn(async () => {
      return {
        ok: true,
        status: 200,
        text: async () =>
          JSON.stringify({
            model: 'local-model',
            choices: [{ message: { content: 'hello' } }],
            usage: { prompt_tokens: 1, completion_tokens: 2, total_tokens: 3 },
          }),
      };
    });

    const llm = new LocalLLMProvider({
      baseUrl: 'http://localhost:11434/v1',
      model: 'local-model',
    });
    const resp = await llm.generate('sys', 'user', { temperature: 0.0 });

    expect(resp.content).toBe('hello');
    expect(resp.modelName).toBe('local-model');
    expect(resp.totalTokens).toBe(3);
    expect((globalThis as any).fetch).toHaveBeenCalledTimes(1);
    expect(((globalThis as any).fetch as any).mock.calls[0][0]).toBe(
      'http://localhost:11434/v1/chat/completions'
    );
  });
});

describe('LocalVisionLLMProvider (OpenAI-compatible)', () => {
  const originalFetch = (globalThis as any).fetch;

  afterEach(() => {
    (globalThis as any).fetch = originalFetch;
  });

  it('should send image_url message content', async () => {
    let capturedBody: any = null;
    (globalThis as any).fetch = jest.fn(async (_url: string, init: any) => {
      capturedBody = JSON.parse(init.body);
      return {
        ok: true,
        status: 200,
        text: async () =>
          JSON.stringify({
            model: 'local-vision',
            choices: [{ message: { content: 'YES' } }],
            usage: { prompt_tokens: 1, completion_tokens: 1, total_tokens: 2 },
          }),
      };
    });

    const llm = new LocalVisionLLMProvider({
      baseUrl: 'http://localhost:1234/v1',
      model: 'local-vision',
    });

    const resp = await llm.generateWithImage('sys', 'is there a button?', 'AAAA', {});
    expect(resp.content).toBe('YES');
    expect(capturedBody.messages[1].content[1].type).toBe('image_url');
    expect(capturedBody.messages[1].content[1].image_url.url).toContain(
      'data:image/png;base64,AAAA'
    );
  });
});

describe('OllamaProvider', () => {
  const originalFetch = (globalThis as any).fetch;

  afterEach(() => {
    (globalThis as any).fetch = originalFetch;
  });

  it('should extend LocalLLMProvider', () => {
    const llm = new OllamaProvider({ model: 'qwen3:8b' });
    expect(llm).toBeInstanceOf(LocalLLMProvider);
  });

  it('should use default base URL http://localhost:11434', () => {
    const llm = new OllamaProvider({ model: 'qwen3:8b' });
    expect(llm.ollamaBaseUrl).toBe('http://localhost:11434');
  });

  it('should accept custom base URL', () => {
    const llm = new OllamaProvider({
      model: 'llama3:8b',
      baseUrl: 'http://192.168.1.100:11434',
    });
    expect(llm.ollamaBaseUrl).toBe('http://192.168.1.100:11434');
  });

  it('should strip trailing slash from base URL', () => {
    const llm = new OllamaProvider({
      model: 'mistral:7b',
      baseUrl: 'http://localhost:11434/',
    });
    expect(llm.ollamaBaseUrl).toBe('http://localhost:11434/');
  });

  it('should report isLocal as true', () => {
    const llm = new OllamaProvider({ model: 'qwen3:4b' });
    expect(llm.isLocal).toBe(true);
  });

  it('should report providerName as ollama', () => {
    const llm = new OllamaProvider({ model: 'phi3:mini' });
    expect(llm.providerName).toBe('ollama');
  });

  it('should report modelName correctly', () => {
    const llm = new OllamaProvider({ model: 'qwen3:8b' });
    expect(llm.modelName).toBe('qwen3:8b');
  });

  it('should return false for supportsJsonMode (conservative default)', () => {
    const llm = new OllamaProvider({ model: 'qwen3:8b' });
    expect(llm.supportsJsonMode()).toBe(false);
  });

  it('should detect vision support for llava models', () => {
    const llm = new OllamaProvider({ model: 'llava:7b' });
    expect(llm.supportsVision()).toBe(true);
  });

  it('should detect vision support for bakllava models', () => {
    const llm = new OllamaProvider({ model: 'bakllava:latest' });
    expect(llm.supportsVision()).toBe(true);
  });

  it('should detect vision support for moondream models', () => {
    const llm = new OllamaProvider({ model: 'moondream:1.8b' });
    expect(llm.supportsVision()).toBe(true);
  });

  it('should return false for vision on text-only models', () => {
    expect(new OllamaProvider({ model: 'qwen3:8b' }).supportsVision()).toBe(false);
    expect(new OllamaProvider({ model: 'llama3:8b' }).supportsVision()).toBe(false);
    expect(new OllamaProvider({ model: 'mistral:7b' }).supportsVision()).toBe(false);
  });

  it('should call /v1/chat/completions endpoint', async () => {
    let capturedUrl: string = '';
    (globalThis as any).fetch = jest.fn(async (url: string) => {
      capturedUrl = url;
      return {
        ok: true,
        status: 200,
        text: async () =>
          JSON.stringify({
            model: 'qwen3:8b',
            choices: [{ message: { content: 'Hello!' } }],
            usage: { prompt_tokens: 5, completion_tokens: 2, total_tokens: 7 },
          }),
      };
    });

    const llm = new OllamaProvider({ model: 'qwen3:8b' });
    const resp = await llm.generate('You are helpful', 'Hi');

    expect(resp.content).toBe('Hello!');
    expect(resp.modelName).toBe('qwen3:8b');
    expect(capturedUrl).toBe('http://localhost:11434/v1/chat/completions');
  });

  it('should work with custom base URL in API calls', async () => {
    let capturedUrl: string = '';
    (globalThis as any).fetch = jest.fn(async (url: string) => {
      capturedUrl = url;
      return {
        ok: true,
        status: 200,
        text: async () =>
          JSON.stringify({
            model: 'llama3:8b',
            choices: [{ message: { content: 'Response' } }],
            usage: { prompt_tokens: 1, completion_tokens: 1, total_tokens: 2 },
          }),
      };
    });

    const llm = new OllamaProvider({
      model: 'llama3:8b',
      baseUrl: 'http://remote-server:11434',
    });
    await llm.generate('sys', 'user');

    expect(capturedUrl).toBe('http://remote-server:11434/v1/chat/completions');
  });
});
