import { test } from 'node:test';
import assert from 'node:assert/strict';

import { splitThinking, withAssistant, withUser } from '../../web/conversation.js';

test('messages are added without changing the history they extend', () => {
  const first = withUser([], 'hi');
  const second = withAssistant(first, 'hello');
  assert.deepEqual(first, [{ role: 'user', content: 'hi' }]);
  assert.deepEqual(second.at(-1), { role: 'assistant', content: 'hello' });
});

test('a reply without reasoning is all answer', () => {
  assert.deepEqual(splitThinking('Four.'), { thinking: null, thinkingDone: true, answer: 'Four.' });
});

test('reasoning still streaming has no answer yet', () => {
  assert.deepEqual(splitThinking('<think>\nAdd th'), { thinking: 'Add th', thinkingDone: false, answer: '' });
});

test('finished reasoning is separated from the answer after it', () => {
  assert.deepEqual(splitThinking('<think>\nAdd them.\n</think>\n\n4'),
    { thinking: 'Add them.', thinkingDone: true, answer: '4' });
});

test('an empty reasoning block, as a template writes when thinking is off, is still reasoning', () => {
  assert.deepEqual(splitThinking('<think>\n\n</think>\n\nHi'), { thinking: '', thinkingDone: true, answer: 'Hi' });
});
