// The composition root. Creates the worker and the views, and wires them
// together; no other module reaches into another's DOM.

import { WorkerClient } from './worker_client.js';
import { showDevice, showStarting, showUnavailable } from './device_status.js';

const deviceStatus = document.querySelector('#device-status');

if (!('gpu' in navigator)) {
  showUnavailable(deviceStatus,
    'This browser does not expose navigator.gpu. Use Chrome or Edge on desktop.');
} else {
  showStarting(deviceStatus);
  const worker = new Worker(`./worker.js${location.search}`, { type: 'module' });
  const client = new WorkerClient(worker, {
    onDevice: (device) => showDevice(deviceStatus, device),
  });
  worker.addEventListener('error', (event) => {
    showDevice(deviceStatus, { ok: false, stage: 'worker', error: event.message });
    client.failAll(new Error(`the worker stopped: ${event.message}`));
  });
}
