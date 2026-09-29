// Axis H. Renders the chat template carried in the model file
// (tokenizer.chat_template, full Jinja2) with the conversation and the
// template variables the model exposes, such as a thinking toggle. It
// renders; it does not interpret.
//
// The whole conversation is rendered every turn. A template may rewrite
// earlier turns — Qwen3's strips past <think> blocks — so the output is not an
// append to the previous one, and the cache's prefix diff decides what is
// reused. The rendered text goes to the tokenizer, which encodes the special
// tokens written in it as their single identifiers.

import { Environment, Interpreter, Template } from './vendor/jinja.js';

export class TemplateError extends Error {
  constructor(message) {
    super(message);
    this.name = 'TemplateError';
  }
}

// `chat` is { template, bosToken, eosToken } as the model file states them.
// Returns render(messages, { variables, now }), a pure function of its
// arguments: `now` is the date for templates that print it.
export function compileTemplate(chat) {
  if (chat.template === null) throw new TemplateError('the model file carries no chat template');
  let parsed;
  try {
    // Template applies the whitespace rules transformers renders with
    // (trim_blocks, lstrip_blocks); only its parse is used.
    parsed = new Template(chat.template).parsed;
  } catch (error) {
    throw new TemplateError(`the model's chat template does not parse: ${error.message}`);
  }
  return (messages, { variables = {}, now }) => {
    const environment = globals(now);
    const context = {
      messages,
      add_generation_prompt: true,
      bos_token: chat.bosToken,
      eos_token: chat.eosToken,
      ...variables,
    };
    try {
      for (const [name, value] of Object.entries(context)) environment.set(name, value);
      return new Interpreter(environment).run(parsed).value;
    } catch (error) {
      if (error instanceof TemplateError) throw error;
      throw new TemplateError(`the model's chat template failed: ${error.message}`);
    }
  };
}

// The names every chat template may use, as transformers defines them.
// Template.render would supply its own, but its strftime_now reads the
// clock and the browser's locale, so the same conversation could render
// differently by date and by language setting. These read only `now`.
function globals(now) {
  const environment = new Environment();
  const constants = { true: true, false: false, none: null, True: true, False: false, None: null };
  for (const [name, value] of Object.entries(constants)) environment.set(name, value);
  environment.set('range', range);
  environment.set('raise_exception', (message) => { throw new TemplateError(message); });
  environment.set('strftime_now', (format) => strftime(now, format));
  return environment;
}

// Python's range().
function range(start, stop, step = 1) {
  if (stop === undefined) [start, stop] = [0, start];
  if (step === 0) throw new TemplateError('range() step must not be zero');
  const values = [];
  for (let i = start; step > 0 ? i < stop : i > stop; i += step) values.push(i);
  return values;
}

// Whether the template reads a thinking switch the interface can offer.
export const offersThinking = (chat) => chat.template?.includes('enable_thinking') ?? false;

const MONTHS = ['Jan', 'Feb', 'Mar', 'Apr', 'May', 'Jun', 'Jul', 'Aug', 'Sep', 'Oct', 'Nov', 'Dec'];

// The strftime directives chat templates use. Any other is an error, not a
// guess.
function strftime(date, format) {
  const pad = (n) => String(n).padStart(2, '0');
  const fields = {
    d: pad(date.getDate()), m: pad(date.getMonth() + 1), Y: String(date.getFullYear()),
    b: MONTHS[date.getMonth()], H: pad(date.getHours()), M: pad(date.getMinutes()),
  };
  return format.replace(/%(.)/g, (_, directive) => {
    if (!(directive in fields)) throw new TemplateError(`unsupported date format %${directive}`);
    return fields[directive];
  });
}
