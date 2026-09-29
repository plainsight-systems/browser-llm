// Axis H: product and interface.
//
// The conversation and its rendering. Holds the messages, sends the rendered
// conversation with the turn's sampling settings and seed through the worker,
// and renders text as it streams back, one message per token.
//
// It holds no cache state. The worker's side diffs each turn against what the
// cache holds, so editing or regenerating an earlier message needs nothing
// from here beyond sending the new conversation.
