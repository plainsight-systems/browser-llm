// Axis H: product and interface.
//
// Renders the chat template carried in the model file (tokenizer.chat_template,
// full Jinja2) with the conversation and the template variables the model's
// policy exposes, such as a thinking toggle. It renders; it does not interpret.
//
// The whole conversation is rendered every turn. A template may rewrite
// earlier turns — Qwen3's strips past <think> blocks — so the output is not an
// append to the previous one, and the cache's prefix diff decides what is
// reused. The rendered text goes to the tokenizer, which encodes the special
// tokens written in it as their single identifiers.
