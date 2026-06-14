// embed.js — shared sentence-embedding integration
// Model: Xenova/paraphrase-multilingual-MiniLM-L12-v2 (same as the chatbot).
//
// Why this is "loaded once": @xenova/transformers caches the model weights in
// the browser's origin-scoped Cache Storage keyed by the model file URLs. So the
// ~120 MB download happens at most once for the whole site, shared between this
// module and the chatbot. The module-level singleton below guarantees a single
// pipeline instance per page. Greek + English both supported (multilingual model).

let _pipe = null;
let _promise = null;
let _failed = false;
const _cache = new Map(); // text -> Float32Array (per page session)

// Load (or reuse) the feature-extraction pipeline. `onProgress` is the optional
// transformers.js progress_callback so the UI can show download progress.
export async function getEmbedder(onProgress) {
  if (_pipe) return _pipe;
  if (_failed) return null;
  if (_promise) return _promise;
  _promise = (async () => {
    try {
      const mod = await import('https://cdn.jsdelivr.net/npm/@xenova/transformers@2.17.2');
      const opts = onProgress ? { progress_callback: onProgress } : undefined;
      const pipe = await mod.pipeline(
        'feature-extraction',
        'Xenova/paraphrase-multilingual-MiniLM-L12-v2',
        opts
      );
      _pipe = pipe;
      return pipe;
    } catch (e) {
      _failed = true;
      return null;
    }
  })();
  return _promise;
}

// Mean-pooled, L2-normalized embedding (matches the chatbot's exact call).
export async function embed(text) {
  const key = String(text || '').trim();
  if (!key) return null;
  if (_cache.has(key)) return _cache.get(key);
  const pipe = await getEmbedder();
  if (!pipe) return null;
  try {
    const out = await pipe(key, { pooling: 'mean', normalize: true });
    const vec = out.data;
    _cache.set(key, vec);
    return vec;
  } catch (e) {
    return null;
  }
}

// Cosine similarity. Inputs are already L2-normalized, so dot product == cosine.
export function cosine(a, b) {
  if (!a || !b || a.length !== b.length) return 0;
  let s = 0;
  for (let i = 0; i < a.length; i++) s += a[i] * b[i];
  return s;
}

export function isFailed() { return _failed; }
export function isReady() { return !!_pipe; }
