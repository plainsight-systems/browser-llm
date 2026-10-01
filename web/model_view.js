// Axis H. The chosen model: what preflight found, and what can be done next —
// download it, load it, remove it — with progress while that happens.
//
// Renders one state object; it holds no state of its own.
//   checking      preflight is running
//   rejected      preflight named what this build cannot run
//   failed        a step failed; `action` names which
//   downloadable  accepted and not cached
//   downloading   `received` of `total` bytes so far
//   cached        accepted and in the cache
//   loading       `done` of `total` bytes handed to the runtime
//   loaded        ready to chat

import { h } from './dom.js';
import { formatBytes } from './units.js';

export function renderModel(element, state, actions) {
  element.dataset.state = TONES[state.phase];
  element.replaceChildren(...CONTENT[state.phase](state, actions));
}

const TONES = {
  checking: 'pending', rejected: 'bad', failed: 'bad', downloadable: 'ok',
  downloading: 'pending', cached: 'ok', loading: 'pending', loaded: 'ok',
};

const CONTENT = {
  checking: ({ model }) => [heading(`Checking ${model.name}…`)],

  rejected: ({ model, verdict }) => [
    heading(`${model.name} cannot run in this build`),
    h('ul', {}, verdict.rejections.map((r) => h('li', { text: r.detail }))),
    facts(verdict),
  ],

  failed: ({ model, action, error }, { retry }) => [
    heading(`Could not ${action} ${model.name}`),
    h('p', { text: error.message }),
    buttons(['Try again', retry, 'primary']),
  ],

  downloadable: ({ model, verdict }, { download }) => [
    heading(`${model.name} can run here`),
    facts(verdict),
    buttons([`Download ${formatBytes(verdict.totalSize)}`, download, 'primary']),
  ],

  downloading: ({ model, received, total }, { cancel }) => [
    heading(`Downloading ${model.name}`),
    progress(received, total),
    buttons(['Cancel', cancel]),
  ],

  cached: ({ model }, { load, remove }) => [
    heading(`${model.name} is downloaded`),
    buttons(['Load', load, 'primary'], ['Remove', remove]),
  ],

  loading: ({ model, done, total }) => [heading(`Loading ${model.name}`), progress(done, total)],

  loaded: ({ model }) => [heading(`${model.name} is loaded`)],
};

const heading = (text) => h('p', { className: 'model-heading', text });

const facts = (verdict) => h('p', { className: 'model-facts',
  text: `${verdict.architecture ?? 'unnamed architecture'} · ${verdict.tensorCount} tensors` });

function progress(done, total) {
  const label = total === null
    ? formatBytes(done)
    : `${formatBytes(done)} of ${formatBytes(total)}`;
  return h('div', { className: 'progress' },
    h('progress', { value: done, max: total ?? undefined, 'aria-label': label }),
    h('span', { text: label }));
}

// Each action is [text, onclick] or [text, onclick, 'primary'] for the step
// the state leads to.
const buttons = (...actions) => h('div', { className: 'actions' },
  actions.map(([text, onclick, kind]) =>
    h('button', { type: 'button', className: kind === 'primary' ? 'action primary' : 'action', text, onclick })));
