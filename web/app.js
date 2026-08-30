// Presentation only. What happened is decided in C++; this renders it.

const statusEl = document.getElementById('status');
const detailEl = document.getElementById('detail');
const adapterEl = document.getElementById('adapter');
const limitsEl = document.getElementById('limits');

const setStatus = (text, cls) => {
  statusEl.textContent = text;
  statusEl.className = cls;
};

const renderPairs = (el, pairs) => {
  el.replaceChildren();
  for (const [key, value] of pairs) {
    const dt = document.createElement('dt');
    dt.textContent = key;
    const dd = document.createElement('dd');
    dd.textContent = value;
    el.append(dt, dd);
  }
};

const mib = (bytes) => `${Number(bytes).toLocaleString()} (${(bytes / 1048576).toFixed(0)} MiB)`;

// Fail early and specifically rather than letting the module fail obscurely.
if (!('gpu' in navigator)) {
  setStatus('WebGPU unavailable', 'bad');
  detailEl.textContent =
    'This browser does not expose navigator.gpu. Chrome or Edge on desktop are the blessed targets.';
} else {
  setStatus('acquiring GPU device…', 'pending');

  const worker = new Worker('./worker.js', { type: 'module' });

  worker.addEventListener('message', ({ data }) => {
    if (data.type === 'ready') {
      setStatus('running self-check…', 'pending');
      return;
    }
    if (data.type !== 'result') return;

    const r = data.result;
    if (!r.ok) {
      setStatus('failed', 'bad');
      detailEl.textContent = `${r.stage ?? 'unknown'}: ${r.error ?? 'no detail reported'}`;
      return;
    }

    setStatus('vector_add: OK', 'ok');
    detailEl.textContent =
      `${r.selfCheck.elements.toLocaleString()} elements computed on the GPU, ` +
      `${r.selfCheck.mismatches} mismatches against the CPU result.`;

    renderPairs(adapterEl, [
      ['description', r.adapter.description || '(not reported)'],
      ['vendor', r.adapter.vendor || '(not reported)'],
      ['architecture', r.adapter.architecture || '(not reported)'],
      ['device', r.adapter.device || '(not reported)'],
      ['backend', r.adapter.backend],
    ]);

    renderPairs(limitsEl, [
      ['maxBufferSize', mib(r.limits.maxBufferSize)],
      ['maxStorageBufferBindingSize', mib(r.limits.maxStorageBufferBindingSize)],
      ['maxComputeWorkgroupsPerDimension', Number(r.limits.maxComputeWorkgroupsPerDimension).toLocaleString()],
      ['maxComputeInvocationsPerWorkgroup', Number(r.limits.maxComputeInvocationsPerWorkgroup).toLocaleString()],
    ]);
  });

  worker.addEventListener('error', (e) => {
    setStatus('failed', 'bad');
    detailEl.textContent = `worker: ${e.message}`;
  });
}
