/**
 * Agent Factory for PlannerExecutorAgent
 *
 * Provides convenient factory functions to create agents with sensible defaults,
 * auto-provider detection, and auto-tracer creation.
 */

import { LLMProvider, OllamaProvider, OpenAIProvider, AnthropicProvider } from '../../llm-provider';
import { createTracer, createLocalTracer, Tracer } from '../../tracing';
import {
  PlannerExecutorConfig,
  ConfigPreset,
  getConfigPreset,
  mergeConfig,
  DEFAULT_CONFIG,
  DeepPartial,
} from './config';

/**
 * Options for creating a PlannerExecutorAgent.
 */
export interface CreateAgentOptions {
  /** Model name for planning (e.g., "gpt-4o", "qwen3:8b") */
  plannerModel: string;

  /** Model name for execution (e.g., "gpt-4o-mini", "qwen3:4b") */
  executorModel: string;

  /** Provider for planner ("auto", "ollama", "openai", "anthropic") */
  plannerProvider?: 'auto' | 'ollama' | 'openai' | 'anthropic';

  /** Provider for executor ("auto", "ollama", "openai", "anthropic") */
  executorProvider?: 'auto' | 'ollama' | 'openai' | 'anthropic';

  /** Ollama server URL (default: http://localhost:11434) */
  ollamaBaseUrl?: string;

  /** OpenAI API key (defaults to OPENAI_API_KEY env var) */
  openaiApiKey?: string;

  /** Anthropic API key (defaults to ANTHROPIC_API_KEY env var) */
  anthropicApiKey?: string;

  /** Configuration preset or partial config */
  config?: ConfigPreset | string | DeepPartial<PlannerExecutorConfig>;

  /** Run ID for tracing (generates UUID if not provided) */
  runId?: string;

  /** Whether to auto-create tracer (default: true) */
  autoTracer?: boolean;
}

/**
 * Auto-detect provider from model name.
 */
export function detectProvider(model: string): 'openai' | 'anthropic' | 'ollama' {
  const modelLower = model.toLowerCase();

  // OpenAI models
  if (
    modelLower.startsWith('gpt-') ||
    modelLower.startsWith('o1-') ||
    modelLower.startsWith('o3-') ||
    modelLower.startsWith('o4-')
  ) {
    return 'openai';
  }

  // Anthropic models
  if (modelLower.startsWith('claude-')) {
    return 'anthropic';
  }

  // Common Ollama model patterns
  const ollamaPatterns = ['qwen', 'llama', 'phi', 'mistral', 'gemma', 'deepseek', 'codellama'];
  if (ollamaPatterns.some(p => modelLower.startsWith(p))) {
    return 'ollama';
  }

  // Ollama models typically have "model:tag" format
  if (model.includes(':')) {
    return 'ollama';
  }

  // Default to ollama for unknown models (assume local)
  return 'ollama';
}

/**
 * Create LLM provider based on provider name.
 */
export function createProvider(
  model: string,
  provider: 'auto' | 'ollama' | 'openai' | 'anthropic',
  options: {
    ollamaBaseUrl?: string;
    openaiApiKey?: string;
    anthropicApiKey?: string;
  }
): LLMProvider {
  const resolvedProvider = provider === 'auto' ? detectProvider(model) : provider;

  switch (resolvedProvider) {
    case 'ollama':
      return new OllamaProvider({
        model,
        baseUrl: options.ollamaBaseUrl ?? 'http://localhost:11434',
      });

    case 'openai': {
      const apiKey = options.openaiApiKey ?? process.env.OPENAI_API_KEY;
      if (!apiKey) {
        throw new Error('OpenAI API key required. Set OPENAI_API_KEY or pass openaiApiKey option.');
      }
      return new OpenAIProvider(apiKey, model);
    }

    case 'anthropic': {
      const apiKey = options.anthropicApiKey ?? process.env.ANTHROPIC_API_KEY;
      if (!apiKey) {
        throw new Error(
          'Anthropic API key required. Set ANTHROPIC_API_KEY or pass anthropicApiKey option.'
        );
      }
      return new AnthropicProvider(apiKey, model);
    }

    default:
      throw new Error(
        `Unknown provider: ${provider}. Supported: 'auto', 'ollama', 'openai', 'anthropic'`
      );
  }
}

/**
 * Resolve configuration from preset or partial config.
 */
export function resolveConfig(
  config?: ConfigPreset | string | DeepPartial<PlannerExecutorConfig>
): PlannerExecutorConfig {
  if (!config) {
    return { ...DEFAULT_CONFIG };
  }

  // String preset name
  if (typeof config === 'string') {
    return getConfigPreset(config);
  }

  // It's a partial config object - merge with defaults
  return mergeConfig(config);
}

/**
 * Result from createPlannerExecutorAgentProviders.
 *
 * Note: The full PlannerExecutorAgent is not yet implemented in TypeScript.
 * This function creates the providers and config that will be used when
 * the agent is ported.
 */
export interface AgentProviders {
  /** Planner LLM provider */
  planner: LLMProvider;

  /** Executor LLM provider */
  executor: LLMProvider;

  /** Resolved configuration */
  config: PlannerExecutorConfig;

  /** Tracer instance (if autoTracer was enabled) */
  tracer?: Tracer;
}

/**
 * Create providers and configuration for PlannerExecutorAgent.
 *
 * This is a helper that creates the LLM providers with auto-detection
 * and resolves configuration from presets. Use this until the full
 * PlannerExecutorAgent is ported to TypeScript.
 *
 * @example Minimal local Ollama setup
 * ```typescript
 * const { planner, executor, config } = await createPlannerExecutorAgentProviders({
 *   plannerModel: 'qwen3:8b',
 *   executorModel: 'qwen3:4b',
 * });
 * ```
 *
 * @example With cloud OpenAI
 * ```typescript
 * const { planner, executor, config } = await createPlannerExecutorAgentProviders({
 *   plannerModel: 'gpt-4o',
 *   executorModel: 'gpt-4o-mini',
 *   openaiApiKey: 'sk-...',
 * });
 * ```
 *
 * @example Mixed cloud planner, local executor
 * ```typescript
 * const { planner, executor, config } = await createPlannerExecutorAgentProviders({
 *   plannerModel: 'gpt-4o',
 *   plannerProvider: 'openai',
 *   executorModel: 'qwen3:4b',
 *   executorProvider: 'ollama',
 *   openaiApiKey: 'sk-...',
 * });
 * ```
 *
 * @example With config preset
 * ```typescript
 * import { ConfigPreset } from '@predicatesystems/runtime';
 *
 * const { planner, executor, config } = await createPlannerExecutorAgentProviders({
 *   plannerModel: 'qwen3:8b',
 *   executorModel: 'qwen3:4b',
 *   config: ConfigPreset.LOCAL_SMALL_MODEL,
 * });
 * ```
 */
export async function createPlannerExecutorAgentProviders(
  options: CreateAgentOptions
): Promise<AgentProviders> {
  const {
    plannerModel,
    executorModel,
    plannerProvider = 'auto',
    executorProvider = 'auto',
    ollamaBaseUrl,
    openaiApiKey,
    anthropicApiKey,
    config,
    runId,
    autoTracer = false,
  } = options;

  // Create providers
  const planner = createProvider(plannerModel, plannerProvider, {
    ollamaBaseUrl,
    openaiApiKey,
    anthropicApiKey,
  });

  const executor = createProvider(executorModel, executorProvider, {
    ollamaBaseUrl,
    openaiApiKey,
    anthropicApiKey,
  });

  // Resolve configuration
  const resolvedConfig = resolveConfig(config);

  // Create tracer if requested
  let tracer: Tracer | undefined;
  if (autoTracer) {
    const apiKey = process.env.PREDICATE_API_KEY;
    if (apiKey) {
      tracer = await createTracer({
        apiKey,
        runId,
        llmModel: `${plannerModel}/${executorModel}`,
        agentType: 'planner-executor',
      });
    } else {
      tracer = createLocalTracer(runId);
    }
  }

  return {
    planner,
    executor,
    config: resolvedConfig,
    tracer,
  };
}
