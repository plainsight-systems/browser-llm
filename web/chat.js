// Axis H: product and interface.
//
// The conversation panel. Holds the messages; renders the whole conversation
// through the model's chat template each turn; sends it with the turn's
// sampling settings and a fresh seed; and renders the reply as it streams.
//
// It holds no cache state. The runtime diffs each turn against what its cache
// holds, so nothing here changes when earlier turns are rewritten.

import { splitThinking, withAssistant, withUser } from './conversation.js';
import { h } from './dom.js';
import { compileTemplate, offersThinking } from './template.js';

const PLACEHOLDER = 'Choose a model to start.';

// `generate(prompt, { sampling, seed, onText })` starts a reply and returns
// { id, reply }; `cancel(id)` stops it early.
export function createChat(root, { generate, cancel }) {
  let active = null;

  const close = () => {
    if (active !== null && active.generating !== null) cancel(active.generating);
    active = null;
    root.replaceChildren(h('p', { className: 'empty', text: PLACEHOLDER }));
  };

  const open = (model, chat) => {
    close();
    let render;
    try {
      render = compileTemplate(chat);
    } catch (error) {
      root.replaceChildren(h('p', { className: 'chat-error', text: error.message }));
      return;
    }
    active = { generating: null };
    root.replaceChildren(conversationPanel(model, chat, render, active, { generate, cancel }));
  };

  close();
  return { open, close };
}

function conversationPanel(model, chat, render, session, { generate, cancel }) {
  let messages = [];

  const log = h('div', { className: 'chat-log', role: 'log', 'aria-live': 'polite' });
  const error = h('p', { className: 'chat-error', 'aria-live': 'assertive' });
  const input = h('textarea', { rows: 2, placeholder: `Message ${model.name}`, 'aria-label': 'Message' });
  const thinking = offersThinking(chat)
    ? h('input', { type: 'checkbox', checked: true, 'aria-label': 'Thinking' })
    : null;
  const button = h('button', { type: 'submit', className: 'action', text: 'Send' });

  const send = async () => {
    const text = input.value.trim();
    if (text === '' || session.generating !== null) return;
    error.textContent = '';

    const mode = thinking?.checked ? 'thinking' : 'default';
    const turn = withUser(messages, text);
    let prompt;
    try {
      const variables = thinking === null ? {} : { enable_thinking: thinking.checked };
      prompt = render(turn, { variables, now: new Date() });
    } catch (failure) {
      error.textContent = failure.message;
      return;
    }

    const userBubble = h('div', { className: 'message user', text });
    const replyBubble = h('div', { className: 'message assistant' });
    log.append(userBubble, replyBubble);
    input.value = '';

    const seed = crypto.getRandomValues(new Uint32Array(1))[0];
    const started = performance.now();
    let reply = '';
    const { id, reply: done } = generate(prompt, {
      sampling: model.policy?.sampling?.[mode],
      seed,
      onText: (piece) => {
        reply += piece;
        renderReply(replyBubble, reply);
        log.scrollTop = log.scrollHeight;
      },
    });
    session.generating = id;
    button.textContent = 'Stop';

    try {
      const { stopReason, tokens } = await done;
      messages = withAssistant(turn, reply);
      const seconds = (performance.now() - started) / 1000;
      replyBubble.append(h('p', { className: 'message-facts',
        text: `${tokens} tokens · ${(tokens / seconds).toFixed(1)} tok/s · seed ${seed}` +
              (stopReason === 'cancelled' ? ' · stopped' : '') }));
    } catch (failure) {
      // The turn did not happen: take it back out, and return the text.
      userBubble.remove();
      replyBubble.remove();
      input.value = text;
      error.textContent = failure.message;
    } finally {
      session.generating = null;
      button.textContent = 'Send';
    }
  };

  const form = h('form', { className: 'composer', onsubmit: (event) => {
    event.preventDefault();
    if (session.generating !== null) cancel(session.generating);
    else send();
  } },
    input,
    h('div', { className: 'composer-row' },
      thinking === null ? h('span') : h('label', { className: 'toggle' }, thinking, ' Thinking'),
      button));

  input.addEventListener('keydown', (event) => {
    if (event.key === 'Enter' && !event.shiftKey) {
      event.preventDefault();
      form.requestSubmit();
    }
  });

  return h('div', { className: 'conversation' },
    h('p', { className: 'chat-heading', text: model.measured ? model.name : `${model.name} · unmeasured` }),
    log, error, form);
}

// A reply's reasoning, collapsed once it is finished, then its answer.
function renderReply(bubble, text) {
  const { thinking, thinkingDone, answer } = splitThinking(text);
  const parts = [];
  if (thinking !== null) {
    parts.push(h('details', { className: 'thinking', open: !thinkingDone },
      h('summary', { text: thinkingDone ? 'Thought' : 'Thinking…' }),
      h('p', { text: thinking })));
  }
  parts.push(h('p', { className: 'answer', text: answer }));
  bubble.replaceChildren(...parts);
}
