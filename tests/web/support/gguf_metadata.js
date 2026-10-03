// Test-only: reads a GGUF file's metadata, its key-value pairs, from the
// start of the file, as tools/gguf_header.py does. The page never needs it:
// the module reads the header (src/core/gguf). Tests use it to take what a
// model file states — its chat template and special tokens — from the
// pinned headers in .cache/test-data/ rather than from a committed copy.

const SCALARS = {
  0: ['getUint8', 1], 1: ['getInt8', 1], 2: ['getUint16', 2], 3: ['getInt16', 2],
  4: ['getUint32', 4], 5: ['getInt32', 4], 6: ['getFloat32', 4], 7: ['getUint8', 1],
  10: ['getBigUint64', 8], 11: ['getBigInt64', 8], 12: ['getFloat64', 8],
};
const STRING = 8;
const ARRAY = 9;

// Returns a Map of key to value; arrays are arrays, 64-bit integers BigInt.
export function readMetadata(bytes) {
  const view = new DataView(bytes.buffer, bytes.byteOffset, bytes.byteLength);
  const decoder = new TextDecoder();
  let at = 0;
  const u32 = () => { const v = view.getUint32(at, true); at += 4; return v; };
  const u64 = () => { const v = Number(view.getBigUint64(at, true)); at += 8; return v; };
  const string = () => {
    const length = u64();
    const text = decoder.decode(bytes.subarray(at, at + length));
    at += length;
    return text;
  };
  const value = (type) => {
    if (type === STRING) return string();
    if (type === ARRAY) {
      const element = u32();
      return Array.from({ length: u64() }, () => value(element));
    }
    const scalar = SCALARS[type];
    if (!scalar) throw new Error(`GGUF value type ${type} at byte ${at} is not one this reader knows`);
    const v = view[scalar[0]](at, true);
    at += scalar[1];
    return type === 7 ? v !== 0 : v;
  };

  if (decoder.decode(bytes.subarray(0, 4)) !== 'GGUF') throw new Error('not a GGUF file');
  at = 8;
  u64();   // tensor count
  const metadata = new Map();
  for (let n = u64(); n > 0; n--) {
    const key = string();
    metadata.set(key, value(u32()));
  }
  return metadata;
}

// The chat a model file states, as web/template.js takes it: its template,
// and its BOS and EOS tokens' text, null where the file declares none.
export function chatOf(metadata) {
  const tokens = metadata.get('tokenizer.ggml.tokens');
  const token = (key) => (metadata.has(key) ? tokens[metadata.get(key)] : null);
  return {
    template: metadata.get('tokenizer.chat_template') ?? null,
    bosToken: token('tokenizer.ggml.bos_token_id'),
    eosToken: token('tokenizer.ggml.eos_token_id'),
  };
}
