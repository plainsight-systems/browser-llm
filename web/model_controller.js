// The chosen model's lifecycle: check it, download it, load it. Holds the one
// state object the model panel renders, and re-renders on every change.
// Choosing another model, or cancelling, abandons the step in progress: its
// work is aborted and any late answer is ignored.

import { cacheKey, downloadModel } from './download.js';
import { fetchRange, openDownload } from './fetch.js';
import { loadModel } from './load.js';
import { renderModel } from './model_view.js';
import { preflight } from './preflight.js';
import { Request } from './protocol.js';
import { requestPersistence } from './opfs.js';

export function createModelController({ element, client, cache, onLoaded, onCacheChanged }) {
  let state = null;
  let step = null;

  const show = (next) => {
    state = next;
    renderModel(element, state, actions);
  };

  // Starts a step, abandoning the one before, and returns its signal.
  const begin = () => {
    step?.abort();
    step = new AbortController();
    return step.signal;
  };

  async function choose(model) {
    const signal = begin();
    show({ phase: 'checking', model });
    try {
      const verdict = await preflight({
        fetchRange: (start, end) => fetchRange(model.url, start, end, { signal }),
        readIndex: (bytes, totalSize) => {
          const copy = bytes.slice().buffer;
          return client.request(Request.PREFLIGHT, { bytes: copy, totalSize }, { transfer: [copy] });
        },
      });
      if (signal.aborted) return;
      if (!verdict.accepted) {
        show({ phase: 'rejected', model, verdict });
        return;
      }
      const cached = (await cache.file(cacheKey(model))) !== null;
      if (!signal.aborted) show({ phase: cached ? 'cached' : 'downloadable', model, verdict });
    } catch (error) {
      if (!signal.aborted) show({ phase: 'failed', model, action: 'check', error });
    }
  }

  async function download() {
    const { model, verdict } = state;
    const signal = begin();
    show({ phase: 'downloading', model, verdict, received: 0, total: model.sizeBytes ?? null });
    requestPersistence();
    try {
      await downloadModel({
        model, cache, signal, openDownload,
        onProgress: (received, total) => {
          if (!signal.aborted) show({ ...state, received, total });
        },
      });
      onCacheChanged();
      if (!signal.aborted) show({ phase: 'cached', model, verdict });
    } catch (error) {
      if (!signal.aborted) show({ phase: 'failed', model, action: 'download', error });
    }
  }

  async function load() {
    const { model, verdict } = state;
    const signal = begin();
    show({ phase: 'loading', model, verdict, done: 0, total: null });
    try {
      const file = await cache.file(cacheKey(model));
      if (file === null) throw new Error('the file is no longer in the cache');
      await loadModel({
        file, signal,
        send: ({ offset, bytes, totalSize }) =>
          client.request(Request.LOAD_CHUNK, { offset, bytes, totalSize }, { transfer: [bytes] }),
        onProgress: (done, total) => {
          if (!signal.aborted) show({ ...state, done, total });
        },
      });
      if (signal.aborted) return;
      show({ phase: 'loaded', model, verdict });
      onLoaded(model);
    } catch (error) {
      if (!signal.aborted) show({ phase: 'failed', model, action: 'load', error });
    }
  }

  async function remove() {
    const { model, verdict } = state;
    await cache.remove(cacheKey(model));
    onCacheChanged();
    show({ phase: 'downloadable', model, verdict });
  }

  const actions = {
    download,
    load,
    remove,
    cancel: () => {
      step?.abort();
      show({ phase: 'downloadable', model: state.model, verdict: state.verdict });
    },
    retry: () => choose(state.model),
  };

  return { choose };
}
