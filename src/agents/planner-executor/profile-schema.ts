/**
 * Zod schemas for profile JSON validation.
 *
 * @see docs/plans/browser-agent/2026-05-02-extensible-categories-and-task-learning.md
 */

import { z } from 'zod';
import { TaskCategory } from './task-category';

export const DataDrivenPruningPolicySchema = z.object({
  allowedRoles: z.array(z.string().min(1)).min(1),
  includeTextPatterns: z.array(z.string()).optional(),
  excludeTextPatterns: z.array(z.string()).optional(),
  maxElements: z.number().int().min(5).max(200),
  maxElementsRelaxed: z.number().int().min(5).max(200),
  maxElementsLoose: z.number().int().min(5).max(200).optional(),
});

export const HeuristicHintInputSchema = z.object({
  intentPattern: z.string().optional(),
  intent_pattern: z.string().optional(),
  textPatterns: z.array(z.string()).optional(),
  text_patterns: z.array(z.string()).optional(),
  roleFilter: z.array(z.string()).optional(),
  role_filter: z.array(z.string()).optional(),
  priority: z.number().int().min(0).optional(),
  attributePatterns: z.record(z.string(), z.string()).optional(),
  attribute_patterns: z.record(z.string(), z.string()).optional(),
});

export const BrowserAgentProfileSchema = z.object({
  id: z.string().min(1),
  label: z.string().min(1),
  version: z.number().int().min(1),
  match: z.object({
    taskKeywords: z.array(z.string()).optional(),
    domainPatterns: z.array(z.string()).optional(),
  }),
  taskCategoryHint: z.nativeEnum(TaskCategory).optional(),
  pruningPolicy: DataDrivenPruningPolicySchema.optional(),
  scoringProfile: z.record(z.string(), z.unknown()).optional(),
  heuristicHints: z.array(HeuristicHintInputSchema).optional(),
  source: z.enum(['built_in', 'user', 'learned', 'imported']),
  priority: z.number().int().optional(),
});

export const BrowserAgentProfileArraySchema = z.array(BrowserAgentProfileSchema);

export const LearnedTargetFingerprintSchema = z.object({
  domain: z.string().min(1),
  taskHash: z.string().min(1),
  intent: z.string().min(1),
  role: z.string().optional(),
  textTokens: z.array(z.string()).optional(),
  ariaTokens: z.array(z.string()).optional(),
  hrefPathPattern: z.string().optional(),
  attributePatterns: z.record(z.string(), z.string()).optional(),
  successCount: z.number().int().min(0),
  failureCount: z.number().int().min(0),
  confidence: z.number().min(0).max(1),
  learnedAt: z.number(),
  lastUsedAt: z.number().optional(),
});

export const DomainProfileSchema = z.object({
  domain: z.string().min(1),
  preferredCategoryHint: z.nativeEnum(TaskCategory).optional(),
  preferredPruningPolicy: DataDrivenPruningPolicySchema.optional(),
  preferredSnapshotLimit: z.number().int().min(5).max(200),
  avgRelaxationLevel: z.number().min(0),
  commonIntents: z.array(z.string()),
  runCount: z.number().int().min(0),
  successRate: z.number().min(0).max(1),
  updatedAt: z.number(),
});
