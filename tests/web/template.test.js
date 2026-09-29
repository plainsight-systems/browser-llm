import { test } from 'node:test';
import assert from 'node:assert/strict';
import { readFile } from 'node:fs/promises';

import { TemplateError, compileTemplate, offersThinking } from '../../web/template.js';

// Each model's real chat template, rendered by Python's jinja2 configured as
// Hugging Face transformers configures it (tools/make_template_fixtures.py).
// The page's engine must produce the same text, byte for byte: the prompt is
// what the model was trained on, and a difference of one newline changes it.
const { fixtures } = JSON.parse(await readFile(new URL('./fixtures/templates.json', import.meta.url)));
const FIXED_DATE = new Date(2026, 8, 29, 12);

for (const { model, chat, cases } of fixtures) {
  test(`${model}: renders every conversation exactly as the reference does`, () => {
    const render = compileTemplate(chat);
    for (const { name, messages, variables, expected } of cases) {
      assert.equal(render(messages, { variables, now: FIXED_DATE }), expected,
        `${name} ${JSON.stringify(variables)}`);
    }
  });
}

test('only the template that reads enable_thinking offers a thinking switch', () => {
  const offered = Object.fromEntries(fixtures.map(({ model, chat }) => [model, offersThinking(chat)]));
  assert.deepEqual(offered, {
    'qwen3-0.6b-q4_0': true,
    'llama-3.2-1b-instruct-q4_0': false,
    'gemma-3-1b-it-q4_0': false,
  });
});

test('a template that raises reports its own message', () => {
  const gemma = fixtures.find(({ model }) => model.startsWith('gemma')).chat;
  const render = compileTemplate(gemma);
  // Gemma's template requires user and assistant turns to alternate.
  assert.throws(() => render([{ role: 'user', content: 'a' }, { role: 'user', content: 'b' }], { now: FIXED_DATE }),
    (error) => error instanceof TemplateError && /alternate/.test(error.message));
});

test('a file without a template, or with one that does not parse, is refused by name', () => {
  assert.throws(() => compileTemplate({ template: null }), /carries no chat template/);
  assert.throws(() => compileTemplate({ template: '{% if %}' }), /does not parse/);
});
