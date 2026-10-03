import { test } from 'node:test';
import assert from 'node:assert/strict';
import { createHash } from 'node:crypto';
import { readFile } from 'node:fs/promises';

import { TemplateError, compileTemplate, offersThinking } from '../../web/template.js';
import { chatOf, readMetadata } from './support/gguf_metadata.js';

// Each model's real chat template, rendered by Python's jinja2 configured as
// Hugging Face transformers configures it (tools/make_template_fixtures.py).
// The page's engine must produce the same text, byte for byte: the prompt is
// what the model was trained on, and a difference of one newline changes it.
//
// The templates are the model publishers' text, so none is committed: each is
// read from its model file's header, pinned by SHA-256 and fetched into
// .cache/test-data/ (make test-data), and the fixture holds the SHA-256 of
// each reference rendering. tools/make_template_fixtures.py --print shows the
// reference text when a hash differs.
const { fixtures } = JSON.parse(await readFile(new URL('./fixtures/templates.json', import.meta.url)));
const FIXED_DATE = new Date(2026, 8, 29, 12);

async function chatFor(model) {
  const path = new URL(`../../.cache/test-data/tokenizer/${model}.header`, import.meta.url);
  const header = await readFile(path).catch((error) => {
    throw new Error(`${model}: no pinned header at ${path.pathname}; run make test-data (${error.code})`);
  });
  return chatOf(readMetadata(header));
}
const chats = Object.fromEntries(await Promise.all(fixtures.map(async ({ model }) => [model, await chatFor(model)])));

const sha256 = (text) => createHash('sha256').update(text, 'utf8').digest('hex');

for (const { model, cases } of fixtures) {
  test(`${model}: renders every conversation exactly as the reference does`, () => {
    const render = compileTemplate(chats[model]);
    for (const { name, messages, variables, sha256: expected } of cases) {
      const rendered = render(messages, { variables, now: FIXED_DATE });
      assert.equal(sha256(rendered), expected, `${name} ${JSON.stringify(variables)} rendered:\n${rendered}`);
    }
  });
}

test('only the template that reads enable_thinking offers a thinking switch', () => {
  const offered = Object.fromEntries(fixtures.map(({ model }) => [model, offersThinking(chats[model])]));
  assert.deepEqual(offered, {
    'qwen3-0.6b-q4_0': true,
    'llama-3.2-1b-instruct-q4_0': false,
    'gemma-3-1b-it-q4_0': false,
  });
});

test('a template that raises reports its own message', () => {
  const gemma = chats[fixtures.find(({ model }) => model.startsWith('gemma')).model];
  const render = compileTemplate(gemma);
  // Gemma's template requires user and assistant turns to alternate.
  assert.throws(() => render([{ role: 'user', content: 'a' }, { role: 'user', content: 'b' }], { now: FIXED_DATE }),
    (error) => error instanceof TemplateError && /alternate/.test(error.message));
});

test('a file without a template, or with one that does not parse, is refused by name', () => {
  assert.throws(() => compileTemplate({ template: null }), /carries no chat template/);
  assert.throws(() => compileTemplate({ template: '{% if %}' }), /does not parse/);
});
