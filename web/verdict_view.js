// Axis H. Shows what preflight found for the chosen model: checking, the
// verdict with every named rejection, or why the check itself failed.

import { h } from './dom.js';

export function showChecking(element, model) {
  element.dataset.state = 'pending';
  element.replaceChildren(h('p', { text: `Checking ${model.name}…` }));
}

export function showVerdict(element, model, verdict) {
  element.dataset.state = verdict.accepted ? 'ok' : 'bad';
  const heading = verdict.accepted
    ? `${model.name} can run here`
    : `${model.name} cannot run in this build`;
  element.replaceChildren(
    h('p', { className: 'verdict-heading', text: heading }),
    h('ul', {}, verdict.rejections.map((r) => h('li', { text: r.detail }))),
    h('p', { className: 'verdict-facts',
      text: `${verdict.architecture ?? 'unnamed architecture'} · ${verdict.tensorCount} tensors` }),
  );
}

export function showFailure(element, model, error) {
  element.dataset.state = 'bad';
  element.replaceChildren(
    h('p', { className: 'verdict-heading', text: `Could not check ${model.name}` }),
    h('p', { text: error.message }),
  );
}
