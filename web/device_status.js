// Axis H. The header line that says whether the GPU is usable, and if not,
// why. Its input is the worker's DEVICE notice.

// Stages at which the runtime itself, rather than the GPU, failed.
const STARTUP_STAGES = new Set(['worker', 'runtime', 'module']);

export function showStarting(element) {
  render(element, 'pending', 'Starting WebGPU…', '');
}

export function showUnavailable(element, reason) {
  render(element, 'bad', 'WebGPU unavailable', reason);
}

export function showDevice(element, device) {
  if (!device.ok) {
    const label = STARTUP_STAGES.has(device.stage) ? 'Runtime failed to start' : 'GPU check failed';
    render(element, 'bad', label, `${device.stage}: ${device.error}`);
    return;
  }
  if (device.fake) {
    render(element, 'bad', 'Fake runtime · no model runs', 'development stand-in selected by ?fake-runtime');
    return;
  }
  if (device.bench !== undefined) {
    console.table(device.bench);
    render(element, 'ok', 'Readback measured', 'results are in the console');
    return;
  }
  const name = device.adapter.description || device.adapter.architecture || 'GPU';
  render(element, 'ok', `${name} · WebGPU`,
    `${device.selfCheck.elements.toLocaleString()} elements checked on the GPU`);
}

function render(element, state, label, detail) {
  element.dataset.state = state;
  element.textContent = label;
  element.title = detail;
}
