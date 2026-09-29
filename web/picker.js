// Axis H: product and interface.
//
// Choosing a model. Lists the curated models and accepts a pasted URL; either
// way, the chosen model is handed to `onChoose`. A model without a measured
// policy is labelled unmeasured: the picker never presents an untested model
// as tuned.

import { h } from './dom.js';
import { CatalogError, modelFromUrl } from './catalog.js';
import { formatBytes } from './units.js';

// Returns { setCached(ids) }, which marks the models whose ids are in `ids`
// as downloaded.
export function createPicker(root, { models, onChoose }) {
  const buttons = new Map();
  const facts = new Map();

  const choose = (model) => {
    for (const [id, button] of buttons) button.setAttribute('aria-pressed', String(id === model.id));
    onChoose(model);
  };

  const list = h('ul', { className: 'models' }, models.map((model) => {
    const fact = h('span', { className: 'model-facts', text: describe(model, false) });
    const button = h('button', { type: 'button', className: 'model', 'aria-pressed': 'false',
      onclick: () => choose(model) },
      h('span', { className: 'model-name', text: model.name }),
      fact);
    buttons.set(model.id, button);
    facts.set(model.id, { model, fact });
    return h('li', {}, button);
  }));

  const urlInput = h('input', { type: 'url', name: 'url', placeholder: 'https://huggingface.co/…/model.gguf',
    'aria-label': 'GGUF file URL', required: true });
  const urlError = h('p', { className: 'field-error', 'aria-live': 'polite' });
  const form = h('form', { className: 'url-form', onsubmit: (event) => {
    event.preventDefault();
    try {
      const model = modelFromUrl(urlInput.value);
      urlError.textContent = '';
      choose(model);
    } catch (error) {
      if (!(error instanceof CatalogError)) throw error;
      urlError.textContent = error.message;
    }
  } },
    h('label', { text: 'Or paste a GGUF URL' }),
    h('div', { className: 'url-row' }, urlInput, h('button', { type: 'submit', text: 'Check' })),
    urlError);

  root.append(list, form);

  return {
    setCached(ids) {
      for (const [id, { model, fact }] of facts) fact.textContent = describe(model, ids.has(id));
    },
  };
}

function describe(model, cached) {
  const size = model.sizeBytes === undefined ? null : formatBytes(model.sizeBytes);
  return [size, cached && 'downloaded', model.measured ? 'measured' : 'unmeasured']
    .filter(Boolean).join(' · ');
}
