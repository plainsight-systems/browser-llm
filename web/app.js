// The composition root. Creates the worker and the views, and wires them
// together; no other module reaches into another's DOM.

import { loadCatalog } from './catalog.js';
import { showDevice, showStarting, showUnavailable } from './device_status.js';
import { fetchRange } from './fetch.js';
import { createPicker } from './picker.js';
import { preflight } from './preflight.js';
import { Request } from './protocol.js';
import * as verdictView from './verdict_view.js';
import { WorkerClient } from './worker_client.js';

const deviceStatus = document.querySelector('#device-status');
const modelsPanel = document.querySelector('#models');
const verdictPanel = document.querySelector('#verdict');

if (!('gpu' in navigator)) {
  showUnavailable(deviceStatus,
    'This browser does not expose navigator.gpu. Use Chrome or Edge on desktop.');
} else {
  const client = startWorker();
  const models = await loadCatalog();
  createPicker(modelsPanel, { models, onChoose: (model) => checkModel(client, model) });
}

function startWorker() {
  showStarting(deviceStatus);
  const worker = new Worker(`./worker.js${location.search}`, { type: 'module' });
  const client = new WorkerClient(worker, {
    onDevice: (device) => showDevice(deviceStatus, device),
  });
  worker.addEventListener('error', (event) => {
    showDevice(deviceStatus, { ok: false, stage: 'worker', error: event.message });
    client.failAll(new Error(`the worker stopped: ${event.message}`));
  });
  return client;
}

// Preflights the chosen model. Choosing another model abandons this check:
// its downloads are aborted and its answer, if one still arrives, is ignored.
let currentCheck = null;

async function checkModel(client, model) {
  currentCheck?.abort();
  const check = new AbortController();
  currentCheck = check;

  verdictView.showChecking(verdictPanel, model);
  try {
    const verdict = await preflight({
      fetchRange: (start, end) => fetchRange(model.url, start, end, { signal: check.signal }),
      readIndex: (bytes, totalSize) => {
        const copy = bytes.slice().buffer;
        return client.request(Request.PREFLIGHT, { bytes: copy, totalSize }, { transfer: [copy] });
      },
    });
    if (!check.signal.aborted) verdictView.showVerdict(verdictPanel, model, verdict);
  } catch (error) {
    if (!check.signal.aborted) verdictView.showFailure(verdictPanel, model, error);
  }
}
