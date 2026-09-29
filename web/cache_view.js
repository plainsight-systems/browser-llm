// Axis H. What the model cache holds, how much space it uses, and whether the
// browser has agreed to keep it. Each cached model can be removed.

import { h } from './dom.js';
import { formatBytes } from './units.js';

export function renderCache(element, { entries, usage, quota, persisted }, { onRemove }) {
  const summary = `${formatBytes(usage)} of ${formatBytes(quota)} used · ` +
    (persisted ? 'kept by the browser' : 'may be cleared by the browser');
  element.replaceChildren(
    h('h2', { text: 'Downloaded' }),
    entries.length === 0
      ? h('p', { className: 'empty', text: 'No models downloaded.' })
      : h('ul', { className: 'cached' }, entries.map((entry) =>
        h('li', {},
          h('span', { text: `${entry.name} · ${formatBytes(entry.sizeBytes)}` }),
          h('button', { type: 'button', className: 'action', text: 'Remove',
            'aria-label': `Remove ${entry.name}`, onclick: () => onRemove(entry.key) })))),
    h('p', { className: 'cache-summary', text: summary }),
  );
}
