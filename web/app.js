// The composition root. Creates the worker, the cache and the views, and
// wires them together; no other module reaches into another's DOM.

import { renderCache } from './cache_view.js';
import { loadCatalog } from './catalog.js';
import { createChat } from './chat.js';
import { showDevice, showStarting, showUnavailable } from './device_status.js';
import { cacheKey } from './download.js';
import { createModelController } from './model_controller.js';
import { ModelCache, storageStatus } from './opfs.js';
import { createPicker } from './picker.js';
import { Request } from './protocol.js';
import { WorkerClient } from './worker_client.js';

const page = {
  deviceStatus: document.querySelector('#device-status'),
  fakeBanner: document.querySelector('#fake-banner'),
  models: document.querySelector('#models'),
  model: document.querySelector('#model'),
  cache: document.querySelector('#cache'),
  chat: document.querySelector('#chat'),
};

if (!('gpu' in navigator)) {
  showUnavailable(page.deviceStatus,
    'This browser does not expose navigator.gpu. Use Chrome or Edge on desktop.');
} else {
  const client = startWorker();
  const [models, cache] = await Promise.all([loadCatalog(), ModelCache.open()]);

  const chat = createChat(page.chat, {
    generate: (prompt, { sampling, seed, onText }) =>
      client.send(Request.GENERATE, { prompt, sampling, seed }, { onToken: onText }),
    cancel: (id) => client.request(Request.CANCEL, { target: id }),
  });

  const picker = createPicker(page.models, {
    models,
    onChoose: (model) => {
      chat.close();
      controller.choose(model);
    },
  });

  const refreshCache = async () => {
    const [entries, status] = await Promise.all([cache.list(), storageStatus()]);
    const cachedKeys = new Set(entries.map((entry) => entry.key));
    picker.setCached(new Set(models.filter((m) => cachedKeys.has(cacheKey(m))).map((m) => m.id)));
    renderCache(page.cache, { entries, ...status }, {
      onRemove: async (key) => {
        await cache.remove(key);
        await refreshCache();
      },
    });
  };

  const controller = createModelController({
    element: page.model,
    client,
    cache,
    onCacheChanged: refreshCache,
    onLoaded: (model, verdict) => chat.open(model, verdict.chat),
  });

  await refreshCache();
}

function startWorker() {
  showStarting(page.deviceStatus);
  const worker = new Worker(`./worker.js${location.search}`, { type: 'module' });
  const client = new WorkerClient(worker, {
    onDevice: (device) => {
      showDevice(page.deviceStatus, device);
      page.fakeBanner.hidden = !device.fake;
    },
  });
  worker.addEventListener('error', (event) => {
    showDevice(page.deviceStatus, { ok: false, stage: 'worker', error: event.message });
    client.failAll(new Error(`the worker stopped: ${event.message}`));
  });
  return client;
}
