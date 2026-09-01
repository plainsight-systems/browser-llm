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

// A failed wgpuAdapterGetInfo and an adapter that legitimately reports an
// empty field are different facts. Rendering both as "(not reported)" would
// hide a query failure behind normal-looking output.
const field = (adapter, name) =>
  adapter.queried === false ? '(query failed)' : (adapter[name] || '(not reported)');

const mib = (bytes) => `${Number(bytes).toLocaleString()} (${(bytes / 1048576).toFixed(0)} MiB)`;

// Fail early and specifically rather than letting the module fail obscurely.
if (!('gpu' in navigator)) {
  setStatus('WebGPU unavailable', 'bad');
  detailEl.textContent =
    'This browser does not expose navigator.gpu. Chrome or Edge on desktop are the blessed targets.';
} else {
  setStatus('acquiring GPU device…', 'pending');

  const worker = new Worker('./worker.js' + location.search, { type: 'module' });

  worker.addEventListener('message', ({ data }) => {
    if (data.type === 'ready') {
      setStatus('running self-check…', 'pending');
      return;
    }
    if (data.type !== 'result') return;

    const r = data.result;
    if (r.bench) {
      const b = r.bench;
      setStatus('readback measured', 'ok');
      detailEl.textContent = `${b.iterations} serialized GPU round trips.`;
      renderPairs(adapterEl, [
        ['sequential min', `${b.seqMinMs.toFixed(3)} ms`],
        ['sequential median', `${b.seqMedianMs.toFixed(3)} ms`],
        ['sequential p95', `${b.seqP95Ms.toFixed(3)} ms`],
        ['sequential max', `${b.seqMaxMs.toFixed(3)} ms`],
        ['sequential mean', `${b.seqMeanMs.toFixed(3)} ms`],
      ]);
      renderPairs(limitsEl, [
        ['ceiling if read back per token', `${(1000 / b.seqMedianMs).toFixed(0)} tok/s`],
        [`batched: ${b.iterations} dispatches, 1 readback`, `${b.batchedTotalMs.toFixed(2)} ms total`],
        ['batched per dispatch', `${(b.batchedTotalMs / b.iterations).toFixed(4)} ms`],
        ['cost of serializing', `${(b.seqMedianMs / (b.batchedTotalMs / b.iterations)).toFixed(0)}x`],
      ]);
      return;
    }

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
      ['description', field(r.adapter, 'description')],
      ['vendor', field(r.adapter, 'vendor')],
      ['architecture', field(r.adapter, 'architecture')],
      ['device', field(r.adapter, 'device')],
      ['backend', field(r.adapter, 'backend')],
    ]);

    renderPairs(limitsEl, [
      ['maxBufferSize (device)', mib(r.limits.maxBufferSize)],
      ['maxStorageBufferBindingSize (device)', mib(r.limits.maxStorageBufferBindingSize)],
      ['maxBufferSize (adapter could grant)', mib(r.adapterMaxima.maxBufferSize)],
      ['maxStorageBufferBindingSize (adapter could grant)', mib(r.adapterMaxima.maxStorageBufferBindingSize)],
      ['maxComputeWorkgroupsPerDimension', Number(r.limits.maxComputeWorkgroupsPerDimension).toLocaleString()],
      ['maxComputeInvocationsPerWorkgroup', Number(r.limits.maxComputeInvocationsPerWorkgroup).toLocaleString()],
      ['maxStorageBuffersPerShaderStage', Number(r.limits.maxStorageBuffersPerShaderStage).toLocaleString()],
      ['minStorageBufferOffsetAlignment', `${Number(r.limits.minStorageBufferOffsetAlignment).toLocaleString()} bytes`],
    ]);
  });

  worker.addEventListener('error', (e) => {
    setStatus('failed', 'bad');
    detailEl.textContent = `worker: ${e.message}`;
  });
}
