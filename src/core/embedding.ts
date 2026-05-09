/**
 * Embedding Service
 *
 * Reads model + dimensions + base URL from `loadConfig()` (config-file +
 * env overrides). Default is OpenAI text-embedding-3-large @ 1536 dims for
 * backward compat.
 *
 * For embedding-asymmetric models (currently the `jina-embeddings-v4`
 * family) the configured model name gets a `-query` or `-passage` suffix
 * appended at request time so a local OpenAI-compatible shim can route
 * prompt_name. OpenAI's symmetric models ignore the task argument.
 *
 * Retry with exponential backoff (4s base, 120s cap, 5 retries).
 * 8000 character input truncation.
 */

import OpenAI from 'openai';
import { loadConfig } from './config.ts';

const DEFAULT_MODEL = 'text-embedding-3-large';
const DEFAULT_DIMENSIONS = 1536;
const MAX_CHARS = 8000;
const MAX_RETRIES = 5;
const BASE_DELAY_MS = 4000;
const MAX_DELAY_MS = 120000;
const BATCH_SIZE = 100;

export type EmbedTask = 'query' | 'passage';

interface Resolved {
  model: string;
  dimensions: number;
  client: OpenAI;
}

let resolved: Resolved | null = null;

function resolve(): Resolved {
  if (resolved) return resolved;
  const config = loadConfig();
  const model = config?.embedding_model ?? DEFAULT_MODEL;
  const dimensions = config?.embedding_dimensions ?? DEFAULT_DIMENSIONS;
  const baseURL = config?.openai_base_url ?? process.env.OPENAI_BASE_URL;
  const apiKey = config?.openai_api_key ?? process.env.OPENAI_API_KEY ?? 'no-auth';
  const client = new OpenAI({
    ...(baseURL ? { baseURL } : {}),
    apiKey,
  });
  resolved = { model, dimensions, client };
  return resolved;
}

// Asymmetric-model families that get a -query / -passage suffix on the model
// name. Local FastAPI shims read the suffix to set prompt_name. Adding a new
// asymmetric family means adding a prefix here.
const ASYMMETRIC_PREFIXES = ['jina-embeddings-v4'];

function modelForTask(baseModel: string, task: EmbedTask): string {
  if (ASYMMETRIC_PREFIXES.some(p => baseModel.startsWith(p))) {
    return `${baseModel}-${task}`;
  }
  return baseModel;
}

export async function embed(
  text: string,
  options: { task?: EmbedTask } = {},
): Promise<Float32Array> {
  const truncated = text.slice(0, MAX_CHARS);
  const result = await embedBatch([truncated], options);
  return result[0];
}

export interface EmbedBatchOptions {
  /**
   * 'query' for search-side embedding, 'passage' for index-side. Defaults
   * to 'passage' so existing callers keep their semantics. Symmetric
   * models (OpenAI tel-3-large) ignore this; asymmetric models (Jina v4)
   * use it to pick prompt_name.
   */
  task?: EmbedTask;
  /**
   * Optional callback fired after each 100-item sub-batch completes.
   * CLI wrappers tick a reporter; Minion handlers can call
   * job.updateProgress here instead of hooking the per-page callback.
   */
  onBatchComplete?: (done: number, total: number) => void;
}

export async function embedBatch(
  texts: string[],
  options: EmbedBatchOptions = {},
): Promise<Float32Array[]> {
  const truncated = texts.map(t => t.slice(0, MAX_CHARS));
  const results: Float32Array[] = [];
  const task: EmbedTask = options.task ?? 'passage';

  // Process in batches of BATCH_SIZE
  for (let i = 0; i < truncated.length; i += BATCH_SIZE) {
    const batch = truncated.slice(i, i + BATCH_SIZE);
    const batchResults = await embedBatchWithRetry(batch, task);
    results.push(...batchResults);
    options.onBatchComplete?.(results.length, truncated.length);
  }

  return results;
}

async function embedBatchWithRetry(
  texts: string[],
  task: EmbedTask,
): Promise<Float32Array[]> {
  const r = resolve();
  const model = modelForTask(r.model, task);
  for (let attempt = 0; attempt < MAX_RETRIES; attempt++) {
    try {
      const response = await r.client.embeddings.create({
        model,
        input: texts,
        dimensions: r.dimensions,
      });

      // Sort by index to maintain order
      const sorted = response.data.sort((a, b) => a.index - b.index);
      return sorted.map(d => new Float32Array(d.embedding));
    } catch (e: unknown) {
      if (attempt === MAX_RETRIES - 1) throw e;

      // Check for rate limit with Retry-After header
      let delay = exponentialDelay(attempt);

      if (e instanceof OpenAI.APIError && e.status === 429) {
        const retryAfter = e.headers?.['retry-after'];
        if (retryAfter) {
          const parsed = parseInt(retryAfter, 10);
          if (!isNaN(parsed)) {
            delay = parsed * 1000;
          }
        }
      }

      await sleep(delay);
    }
  }

  // Should not reach here
  throw new Error('Embedding failed after all retries');
}

function exponentialDelay(attempt: number): number {
  const delay = BASE_DELAY_MS * Math.pow(2, attempt);
  return Math.min(delay, MAX_DELAY_MS);
}

function sleep(ms: number): Promise<void> {
  return new Promise(resolve => setTimeout(resolve, ms));
}

export function getEmbeddingModel(): string {
  return resolve().model;
}
export function getEmbeddingDimensions(): number {
  return resolve().dimensions;
}

// Legacy const exports — frozen at import time; do not use in code that
// runs after config is loaded. Prefer the function form above. Kept so
// existing imports keep compiling.
export const EMBEDDING_MODEL = DEFAULT_MODEL;
export const EMBEDDING_DIMENSIONS = DEFAULT_DIMENSIONS;
