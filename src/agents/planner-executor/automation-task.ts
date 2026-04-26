import { z } from 'zod';
import { TaskCategory, normalizeTaskCategory } from './task-category';

export interface AutomationTask {
  task: string;
  startUrl?: string;
  goal?: Record<string, unknown>;
  category?: TaskCategory | null;
  domainHints: string[];
}

const TaskCategorySchema = z.preprocess(value => {
  if (typeof value === 'string') {
    return normalizeTaskCategory(value);
  }
  return value;
}, z.nativeEnum(TaskCategory).nullable().optional());

export const AutomationTaskSchema = z.object({
  task: z.string().min(1),
  startUrl: z.string().url().optional(),
  goal: z.record(z.string(), z.unknown()).optional(),
  category: TaskCategorySchema,
  domainHints: z.array(z.string()).default([]),
});
