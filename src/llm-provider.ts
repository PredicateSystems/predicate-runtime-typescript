/**
 * LLM Provider Abstraction - BYOB (Bring Your Own Brain)
 * Enables pluggable LLM support for SentienceAgent
 */

/**
 * Response from LLM provider
 */
export interface LLMResponse {
  content: string;
  promptTokens?: number;
  completionTokens?: number;
  totalTokens?: number;
  modelName?: string;
}

/**
 * Abstract base class for LLM providers
 * Implement this to integrate any LLM (OpenAI, Anthropic, Local, etc.)
 */
export abstract class LLMProvider {
  /**
   * Generate LLM response from prompts
   * @param systemPrompt - System/instruction prompt
   * @param userPrompt - User query prompt
   * @param options - Additional provider-specific options
   */
  abstract generate(
    systemPrompt: string,
    userPrompt: string,

    options?: Record<string, any>
  ): Promise<LLMResponse>;

  /**
   * Whether this provider supports JSON mode (structured output)
   */
  abstract supportsJsonMode(): boolean;

  /**
   * Get the model name/identifier
   */
  abstract get modelName(): string;

  /**
   * Whether this provider supports image input for vision tasks.
   * Override in subclasses that support vision-capable models.
   */
  supportsVision(): boolean {
    return false;
  }

  /**
   * Generate with image input (vision-capable models only).
   * Override in subclasses that support vision.
   */

  generateWithImage(
    _systemPrompt: string,
    _userPrompt: string,
    _imageBase64: string,
    _options: Record<string, any> = {}
  ): Promise<LLMResponse> {
    throw new Error(
      `${this.constructor.name} does not support vision. ` +
        `Use a vision-capable provider (e.g., OpenAIProvider with GPT-4o, AnthropicProvider with Claude 3).`
    );
  }
}

/**
 * Local OpenAI-compatible Provider (Ollama / LM Studio / llama.cpp server, etc.)
 *
 * This is the TypeScript equivalent of Python's LocalLLMProvider concept, but instead of
 * embedding a full HF runtime inside Node, it calls a local HTTP server that exposes an
 * OpenAI-compatible Chat Completions API.
 *
 * Examples of compatible local servers:
 * - Ollama (OpenAI-compatible endpoint)
 * - LM Studio (OpenAI-compatible endpoint)
 * - llama.cpp server (OpenAI-compatible endpoint)
 */
export class LocalLLMProvider extends LLMProvider {
  private _modelName: string;
  private _baseUrl: string;
  private _apiKey?: string;
  private _defaultHeaders: Record<string, string>;
  private _timeoutMs: number;

  constructor(
    options: {
      model?: string;
      baseUrl?: string;
      apiKey?: string;
      timeoutMs?: number;
      headers?: Record<string, string>;
    } = {}
  ) {
    super();
    this._modelName = options.model ?? process.env.SENTIENCE_LOCAL_LLM_MODEL ?? 'local-model';
    // Common defaults:
    // - Ollama OpenAI-compatible: http://localhost:11434/v1
    // - LM Studio: http://localhost:1234/v1
    this._baseUrl =
      options.baseUrl ?? process.env.SENTIENCE_LOCAL_LLM_BASE_URL ?? 'http://localhost:11434/v1';
    this._apiKey = options.apiKey ?? process.env.SENTIENCE_LOCAL_LLM_API_KEY;
    this._timeoutMs = options.timeoutMs ?? 60_000;
    this._defaultHeaders = {
      'Content-Type': 'application/json',
      ...(options.headers ?? {}),
    };
    if (this._apiKey) {
      this._defaultHeaders.Authorization = `Bearer ${this._apiKey}`;
    }
  }

  supportsJsonMode(): boolean {
    // Many local OpenAI-compatible servers don't reliably implement response_format=json_object.
    return false;
  }

  get modelName(): string {
    return this._modelName;
  }

  async generate(
    systemPrompt: string,
    userPrompt: string,
    options: Record<string, any> = {}
  ): Promise<LLMResponse> {
    const fetchFn = (globalThis as any).fetch as typeof fetch | undefined;
    if (!fetchFn) {
      throw new Error(
        'Global fetch is not available. Use Node 18+ or polyfill fetch before using LocalLLMProvider.'
      );
    }

    const controller = new AbortController();
    const timeoutId = setTimeout(() => controller.abort(), options.timeoutMs ?? this._timeoutMs);

    const payload: any = {
      model: this._modelName,
      messages: [
        ...(systemPrompt ? [{ role: 'system', content: systemPrompt }] : []),
        { role: 'user', content: userPrompt },
      ],
      temperature: options.temperature ?? 0.0,
    };

    if (options.max_tokens !== undefined) payload.max_tokens = options.max_tokens;
    if (options.top_p !== undefined) payload.top_p = options.top_p;

    // Allow pass-through of server-specific fields, but avoid overriding core fields accidentally
    const { timeoutMs: _ignoredTimeout, ...rest } = options;
    Object.assign(payload, rest);

    try {
      const res = await fetchFn(`${this._baseUrl}/chat/completions`, {
        method: 'POST',
        headers: this._defaultHeaders,
        body: JSON.stringify(payload),
        signal: controller.signal,
      });

      const text = await res.text();
      if (!res.ok) {
        throw new Error(`Local LLM HTTP ${res.status}: ${text.slice(0, 500)}`);
      }

      const data = JSON.parse(text);
      const choice = data?.choices?.[0];
      const message = choice?.message;

      // Extract content - Ollama/Qwen3 sometimes puts output in 'reasoning' field
      // when the model uses thinking mode
      let content = message?.content ?? '';

      // If content is empty but reasoning exists, try to extract the answer from reasoning
      // The reasoning field often ends with the actual answer after the thinking
      if (!content && message?.reasoning) {
        const reasoningLen = message.reasoning.length;
        // Debug: log the END of the reasoning field (where the answer usually is)
        const reasoningEnd = message.reasoning.slice(-300);
        console.log(
          `[LocalLLMProvider DEBUG] Empty content, reasoning len=${reasoningLen}, last 300 chars:\n${reasoningEnd}`
        );

        // First, try to extract JSON objects (for planner responses)
        // Look for the LAST complete JSON object in the reasoning
        const jsonMatches = [
          ...message.reasoning.matchAll(/\{[^{}]*"action"\s*:\s*"[^"]+"\s*[^{}]*\}/g),
        ];
        if (jsonMatches.length > 0) {
          content = jsonMatches[jsonMatches.length - 1][0];
          console.log(`[LocalLLMProvider DEBUG] Extracted JSON from reasoning: ${content}`);
        } else {
          // Look for the LAST action pattern in reasoning (answer comes after thinking)
          // Use matchAll to find all occurrences and take the last one
          // Priority: TYPE > CLICK > DONE > NONE (more specific actions first)
          const typeMatches = [...message.reasoning.matchAll(/TYPE\(\d+,\s*"[^"]*"\)/g)];
          if (typeMatches.length > 0) {
            content = typeMatches[typeMatches.length - 1][0];
            console.log(`[LocalLLMProvider DEBUG] Extracted TYPE from reasoning: ${content}`);
          } else {
            const clickMatches = [...message.reasoning.matchAll(/CLICK\(\d+\)/g)];
            if (clickMatches.length > 0) {
              content = clickMatches[clickMatches.length - 1][0];
              console.log(`[LocalLLMProvider DEBUG] Extracted CLICK from reasoning: ${content}`);
            } else if (message.reasoning.includes('DONE')) {
              // Only use DONE if it's near the end (within last 100 chars)
              if (message.reasoning.slice(-100).includes('DONE')) {
                content = 'DONE';
                console.log(`[LocalLLMProvider DEBUG] Extracted DONE from end of reasoning`);
              }
            }
            // Don't extract NONE - if model is still reasoning, let it continue
          }
        }

        // Final fallback: if no structured pattern matched, use the last
        // non-empty line of reasoning as the content. This handles plain text
        // answers from extraction tasks where the model thinks but doesn't
        // produce content outside the reasoning field.
        if (!content) {
          const lines = message.reasoning.split('\n');
          for (let i = lines.length - 1; i >= 0; i--) {
            const line = lines[i].trim();
            if (line && !line.startsWith('```') && !line.startsWith('//')) {
              content = line;
              console.log(
                `[LocalLLMProvider DEBUG] Extracted last reasoning line as content: ${content.slice(0, 100)}`
              );
              break;
            }
          }
        }
      }

      const usage = data?.usage;

      return {
        content,
        promptTokens: usage?.prompt_tokens,
        completionTokens: usage?.completion_tokens,
        totalTokens: usage?.total_tokens,
        modelName: data?.model ?? this._modelName,
      };
    } finally {
      clearTimeout(timeoutId);
    }
  }
}

/**
 * Local OpenAI-compatible Vision Provider.
 *
 * This is the TypeScript analogue of Python's LocalVisionLLMProvider, but it assumes your
 * local server supports the OpenAI vision message format (image_url with data: URI).
 *
 * If your local stack uses a different schema (e.g., Ollama images array), you can implement
 * a custom provider by extending LLMProvider.
 */
export class LocalVisionLLMProvider extends LocalLLMProvider {
  supportsVision(): boolean {
    return true;
  }

  async generateWithImage(
    systemPrompt: string,
    userPrompt: string,
    imageBase64: string,
    options: Record<string, any> = {}
  ): Promise<LLMResponse> {
    const fetchFn = (globalThis as any).fetch as typeof fetch | undefined;
    if (!fetchFn) {
      throw new Error(
        'Global fetch is not available. Use Node 18+ or polyfill fetch before using LocalVisionLLMProvider.'
      );
    }

    const controller = new AbortController();
    const timeoutId = setTimeout(
      () => controller.abort(),
      options.timeoutMs ?? (this as any)._timeoutMs ?? 60_000
    );

    // Rebuild payload (we cannot reuse LocalLLMProvider.generate because message shape differs)
    const modelName = (this as any)._modelName ?? 'local-model';
    const baseUrl = (this as any)._baseUrl ?? 'http://localhost:11434/v1';
    const headers = (this as any)._defaultHeaders ?? { 'Content-Type': 'application/json' };

    const payload: any = {
      model: modelName,
      messages: [
        ...(systemPrompt ? [{ role: 'system', content: systemPrompt }] : []),
        {
          role: 'user',
          content: [
            { type: 'text', text: userPrompt },
            { type: 'image_url', image_url: { url: `data:image/png;base64,${imageBase64}` } },
          ],
        },
      ],
      temperature: options.temperature ?? 0.0,
    };

    if (options.max_tokens !== undefined) payload.max_tokens = options.max_tokens;
    const { timeoutMs: _ignoredTimeout, ...rest } = options;
    Object.assign(payload, rest);

    try {
      const res = await fetchFn(`${baseUrl}/chat/completions`, {
        method: 'POST',
        headers,
        body: JSON.stringify(payload),
        signal: controller.signal,
      });

      const text = await res.text();
      if (!res.ok) {
        throw new Error(`Local Vision LLM HTTP ${res.status}: ${text.slice(0, 500)}`);
      }

      const data = JSON.parse(text);
      const choice = data?.choices?.[0];
      const content = choice?.message?.content ?? '';
      const usage = data?.usage;

      return {
        content,
        promptTokens: usage?.prompt_tokens,
        completionTokens: usage?.completion_tokens,
        totalTokens: usage?.total_tokens,
        modelName: data?.model ?? modelName,
      };
    } finally {
      clearTimeout(timeoutId);
    }
  }
}

/**
 * Ollama Provider - dedicated wrapper for Ollama local LLM server.
 *
 * Ollama serves models locally and provides an OpenAI-compatible endpoint at /v1.
 * This provider wraps LocalLLMProvider with sensible defaults for Ollama.
 *
 * @example
 * ```typescript
 * import { OllamaProvider } from '@predicatesystems/runtime';
 *
 * const llm = new OllamaProvider({ model: 'qwen3:8b' });
 * const response = await llm.generate('You are helpful', 'Hello!');
 * console.log(response.content);
 * ```
 *
 * @example Custom base URL
 * ```typescript
 * const llm = new OllamaProvider({
 *   model: 'llama3:8b',
 *   baseUrl: 'http://192.168.1.100:11434'
 * });
 * ```
 */
export class OllamaProvider extends LocalLLMProvider {
  private _ollamaBaseUrl: string;
  private _ollamaModelName: string;
  private _disableThinking: boolean;

  constructor(
    options: { model: string; baseUrl?: string; timeoutMs?: number; disableThinking?: boolean } = {
      model: 'qwen3:8b',
    }
  ) {
    const baseUrl = options.baseUrl ?? 'http://localhost:11434';
    // Ollama serves OpenAI-compatible API at /v1
    super({
      model: options.model,
      baseUrl: `${baseUrl.replace(/\/$/, '')}/v1`,
      apiKey: 'ollama', // Ollama doesn't require a real API key
      timeoutMs: options.timeoutMs,
    });
    this._ollamaBaseUrl = baseUrl;
    this._ollamaModelName = options.model;
    // For Qwen3 and similar models, disable thinking by default if model name contains "qwen"
    this._disableThinking = options.disableThinking ?? options.model.toLowerCase().includes('qwen');
  }

  /**
   * Override generate to add Ollama-specific options for Qwen3 thinking mode.
   */
  async generate(
    systemPrompt: string,
    userPrompt: string,
    options: Record<string, any> = {}
  ): Promise<LLMResponse> {
    // For Qwen3 models, add think: false to disable reasoning output
    // Ollama OpenAI-compatible API expects 'think' as a TOP-LEVEL field,
    // NOT nested under 'options'. See: https://github.com/ollama/ollama/blob/main/docs/openai.md
    const ollamaOptions = { ...options };
    if (this._disableThinking) {
      ollamaOptions.think = false;
    }
    return super.generate(systemPrompt, userPrompt, ollamaOptions);
  }

  /**
   * Ollama runs locally.
   */
  get isLocal(): boolean {
    return true;
  }

  /**
   * Provider identifier.
   */
  get providerName(): string {
    return 'ollama';
  }

  /**
   * Get the Ollama base URL (without /v1 suffix).
   */
  get ollamaBaseUrl(): string {
    return this._ollamaBaseUrl;
  }

  /**
   * JSON mode support varies by Ollama model.
   * Most instruction-tuned models (qwen, llama, mistral) can output JSON
   * with proper prompting, but native JSON mode is model-dependent.
   */
  supportsJsonMode(): boolean {
    // Conservative default: rely on prompt engineering for JSON
    return false;
  }

  /**
   * Vision support varies by Ollama model.
   * Models like llava, bakllava, moondream support vision.
   */
  supportsVision(): boolean {
    const modelLower = this._ollamaModelName.toLowerCase();
    return ['llava', 'bakllava', 'moondream'].some(v => modelLower.includes(v));
  }
}

/**
 * OpenAI Provider (GPT-4, GPT-4o, etc.)
 * Requires: npm install openai
 */
export class OpenAIProvider extends LLMProvider {
  private client: any;
  private _modelName: string;

  constructor(apiKey: string, model: string = 'gpt-4o') {
    super();

    // Lazy import to avoid requiring openai package if not used
    try {
      // eslint-disable-next-line @typescript-eslint/no-require-imports
      const { OpenAI } = require('openai');
      this.client = new OpenAI({ apiKey });
    } catch {
      throw new Error('OpenAI package not installed. Run: npm install openai');
    }

    this._modelName = model;
  }

  async generate(
    systemPrompt: string,
    userPrompt: string,
    options: Record<string, any> = {}
  ): Promise<LLMResponse> {
    const response = await this.client.chat.completions.create({
      model: this._modelName,
      messages: [
        { role: 'system', content: systemPrompt },
        { role: 'user', content: userPrompt },
      ],
      temperature: options.temperature ?? 0.0,
      ...options,
    });

    const choice = response.choices[0];
    return {
      content: choice.message.content || '',
      promptTokens: response.usage?.prompt_tokens,
      completionTokens: response.usage?.completion_tokens,
      totalTokens: response.usage?.total_tokens,
      modelName: this._modelName,
    };
  }

  supportsJsonMode(): boolean {
    return true;
  }

  supportsVision(): boolean {
    return true;
  }

  async generateWithImage(
    systemPrompt: string,
    userPrompt: string,
    imageBase64: string,
    options: Record<string, any> = {}
  ): Promise<LLMResponse> {
    const response = await this.client.chat.completions.create({
      model: this._modelName,
      messages: [
        { role: 'system', content: systemPrompt },
        {
          role: 'user',
          content: [
            { type: 'text', text: userPrompt },
            { type: 'image_url', image_url: { url: `data:image/png;base64,${imageBase64}` } },
          ],
        },
      ],
      temperature: options.temperature ?? 0.0,
      ...options,
    });

    const choice = response.choices[0];
    return {
      content: choice.message.content || '',
      promptTokens: response.usage?.prompt_tokens,
      completionTokens: response.usage?.completion_tokens,
      totalTokens: response.usage?.total_tokens,
      modelName: this._modelName,
    };
  }

  get modelName(): string {
    return this._modelName;
  }
}

/**
 * Anthropic Provider (Claude 3.5 Sonnet, etc.)
 * Requires: npm install @anthropic-ai/sdk
 */
export class AnthropicProvider extends LLMProvider {
  private client: any;
  private _modelName: string;

  constructor(apiKey: string, model: string = 'claude-3-5-sonnet-20241022') {
    super();

    try {
      // eslint-disable-next-line @typescript-eslint/no-require-imports
      const { Anthropic } = require('@anthropic-ai/sdk');
      this.client = new Anthropic({ apiKey });
    } catch {
      throw new Error('Anthropic SDK not installed. Run: npm install @anthropic-ai/sdk');
    }

    this._modelName = model;
  }

  async generate(
    systemPrompt: string,
    userPrompt: string,
    options: Record<string, any> = {}
  ): Promise<LLMResponse> {
    const response = await this.client.messages.create({
      model: this._modelName,
      max_tokens: options.max_tokens ?? 1024,
      system: systemPrompt,
      messages: [{ role: 'user', content: userPrompt }],
      temperature: options.temperature ?? 0.0,
      ...options,
    });

    const content = response.content[0].text;
    return {
      content,
      promptTokens: response.usage?.input_tokens,
      completionTokens: response.usage?.output_tokens,
      totalTokens: (response.usage?.input_tokens || 0) + (response.usage?.output_tokens || 0),
      modelName: this._modelName,
    };
  }

  supportsJsonMode(): boolean {
    // Claude supports structured output but not via "json_mode" flag
    return false;
  }

  supportsVision(): boolean {
    return true;
  }

  async generateWithImage(
    systemPrompt: string,
    userPrompt: string,
    imageBase64: string,
    options: Record<string, any> = {}
  ): Promise<LLMResponse> {
    const response = await this.client.messages.create({
      model: this._modelName,
      max_tokens: options.max_tokens ?? 1024,
      system: systemPrompt,
      messages: [
        {
          role: 'user',
          content: [
            { type: 'text', text: userPrompt },
            {
              type: 'image',
              source: {
                type: 'base64',
                media_type: options.media_type ?? 'image/png',
                data: imageBase64,
              },
            },
          ],
        },
      ],
      temperature: options.temperature ?? 0.0,
      ...options,
    });

    const content = response.content[0].text;
    return {
      content,
      promptTokens: response.usage?.input_tokens,
      completionTokens: response.usage?.output_tokens,
      totalTokens: (response.usage?.input_tokens || 0) + (response.usage?.output_tokens || 0),
      modelName: this._modelName,
    };
  }

  get modelName(): string {
    return this._modelName;
  }
}

/**
 * Zhipu AI GLM Provider (GLM-4, GLM-4-Plus, etc.)
 *
 * Requirements:
 *   npm install zhipuai-sdk-nodejs-v4
 */
export class GLMProvider extends LLMProvider {
  private client: any;
  private _modelName: string;

  constructor(apiKey: string, model: string = 'glm-4-plus') {
    super();

    try {
      // eslint-disable-next-line @typescript-eslint/no-require-imports
      const ZhipuAI = require('zhipuai-sdk-nodejs-v4');
      this.client = new ZhipuAI({ apiKey });
    } catch {
      throw new Error('ZhipuAI SDK not installed. Run: npm install zhipuai-sdk-nodejs-v4');
    }

    this._modelName = model;
  }

  async generate(
    systemPrompt: string,
    userPrompt: string,
    options: Record<string, any> = {}
  ): Promise<LLMResponse> {
    const messages: any[] = [];
    if (systemPrompt) {
      messages.push({ role: 'system', content: systemPrompt });
    }
    messages.push({ role: 'user', content: userPrompt });

    const response = await this.client.chat.completions.create({
      model: this._modelName,
      messages,
      temperature: options.temperature ?? 0.0,
      max_tokens: options.max_tokens,
      ...options,
    });

    const choice = response.choices[0];
    const usage = response.usage;

    return {
      content: choice.message?.content || '',
      promptTokens: usage?.prompt_tokens,
      completionTokens: usage?.completion_tokens,
      totalTokens: usage?.total_tokens,
      modelName: this._modelName,
    };
  }

  supportsJsonMode(): boolean {
    // GLM-4 models support JSON mode
    return this._modelName.toLowerCase().includes('glm-4');
  }

  get modelName(): string {
    return this._modelName;
  }
}

/**
 * Google Gemini Provider (Gemini 2.0, Gemini 1.5 Pro, etc.)
 *
 * Requirements:
 *   npm install @google/generative-ai
 */
export class GeminiProvider extends LLMProvider {
  private model: any;
  private _modelName: string;

  constructor(apiKey: string, model: string = 'gemini-2.0-flash-exp') {
    super();

    try {
      // eslint-disable-next-line @typescript-eslint/no-require-imports
      const { GoogleGenerativeAI } = require('@google/generative-ai');
      const genAI = new GoogleGenerativeAI(apiKey);
      this.model = genAI.getGenerativeModel({ model });
    } catch {
      throw new Error(
        'Google Generative AI SDK not installed. Run: npm install @google/generative-ai'
      );
    }

    this._modelName = model;
  }

  async generate(
    systemPrompt: string,
    userPrompt: string,
    options: Record<string, any> = {}
  ): Promise<LLMResponse> {
    // Combine system and user prompts (Gemini doesn't have separate system role in all versions)
    const fullPrompt = systemPrompt ? `${systemPrompt}\n\n${userPrompt}` : userPrompt;

    // Build generation config
    const generationConfig: any = {
      temperature: options.temperature ?? 0.0,
    };

    if (options.max_tokens) {
      generationConfig.maxOutputTokens = options.max_tokens;
    }

    // Merge additional parameters
    Object.assign(generationConfig, options);

    // Call Gemini API
    const result = await this.model.generateContent({
      contents: [{ role: 'user', parts: [{ text: fullPrompt }] }],
      generationConfig,
    });

    const response = result.response;
    const content = response.text() || '';

    // Extract token usage if available
    let promptTokens: number | undefined;
    let completionTokens: number | undefined;
    let totalTokens: number | undefined;

    if (response.usageMetadata) {
      promptTokens = response.usageMetadata.promptTokenCount;
      completionTokens = response.usageMetadata.candidatesTokenCount;
      totalTokens = response.usageMetadata.totalTokenCount;
    }

    return {
      content,
      promptTokens,
      completionTokens,
      totalTokens,
      modelName: this._modelName,
    };
  }

  supportsJsonMode(): boolean {
    // Gemini 1.5+ models support JSON mode via response_mime_type
    const modelLower = this._modelName.toLowerCase();
    return modelLower.includes('gemini-1.5') || modelLower.includes('gemini-2.0');
  }

  get modelName(): string {
    return this._modelName;
  }
}
