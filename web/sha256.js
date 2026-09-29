// Axis L. SHA-256 over a stream, so a downloaded file is verified as it
// arrives. The browser's crypto.subtle.digest takes the whole input at once,
// which for a model means holding the whole file in memory.

const K = new Uint32Array([
  0x428a2f98, 0x71374491, 0xb5c0fbcf, 0xe9b5dba5, 0x3956c25b, 0x59f111f1, 0x923f82a4, 0xab1c5ed5,
  0xd807aa98, 0x12835b01, 0x243185be, 0x550c7dc3, 0x72be5d74, 0x80deb1fe, 0x9bdc06a7, 0xc19bf174,
  0xe49b69c1, 0xefbe4786, 0x0fc19dc6, 0x240ca1cc, 0x2de92c6f, 0x4a7484aa, 0x5cb0a9dc, 0x76f988da,
  0x983e5152, 0xa831c66d, 0xb00327c8, 0xbf597fc7, 0xc6e00bf3, 0xd5a79147, 0x06ca6351, 0x14292967,
  0x27b70a85, 0x2e1b2138, 0x4d2c6dfc, 0x53380d13, 0x650a7354, 0x766a0abb, 0x81c2c92e, 0x92722c85,
  0xa2bfe8a1, 0xa81a664b, 0xc24b8b70, 0xc76c51a3, 0xd192e819, 0xd6990624, 0xf40e3585, 0x106aa070,
  0x19a4c116, 0x1e376c08, 0x2748774c, 0x34b0bcb5, 0x391c0cb3, 0x4ed8aa4a, 0x5b9cca4f, 0x682e6ff3,
  0x748f82ee, 0x78a5636f, 0x84c87814, 0x8cc70208, 0x90befffa, 0xa4506ceb, 0xbef9a3f7, 0xc67178f2,
]);

const rotr = (x, n) => (x >>> n) | (x << (32 - n));

export class Sha256 {
  #state = new Uint32Array([
    0x6a09e667, 0xbb67ae85, 0x3c6ef372, 0xa54ff53a, 0x510e527f, 0x9b05688c, 0x1f83d9ab, 0x5be0cd19,
  ]);
  #block = new Uint8Array(64);
  #blockLength = 0;
  #totalBytes = 0;
  #words = new Uint32Array(64);

  update(bytes) {
    let offset = 0;
    this.#totalBytes += bytes.length;
    // Top up a partial block first, then compress whole blocks straight from
    // the input, and keep what is left for the next call.
    if (this.#blockLength > 0) {
      const take = Math.min(64 - this.#blockLength, bytes.length);
      this.#block.set(bytes.subarray(0, take), this.#blockLength);
      this.#blockLength += take;
      offset = take;
      if (this.#blockLength < 64) return this;
      this.#compress(this.#block, 0);
      this.#blockLength = 0;
    }
    for (; offset + 64 <= bytes.length; offset += 64) this.#compress(bytes, offset);
    this.#block.set(bytes.subarray(offset), 0);
    this.#blockLength = bytes.length - offset;
    return this;
  }

  // The digest as lowercase hex. The hash cannot be updated afterwards.
  hex() {
    const bitLength = this.#totalBytes * 8;
    const padding = new Uint8Array(((this.#blockLength < 56 ? 56 : 120) - this.#blockLength) + 8);
    padding[0] = 0x80;
    const view = new DataView(padding.buffer);
    view.setUint32(padding.length - 8, Math.floor(bitLength / 2 ** 32));
    view.setUint32(padding.length - 4, bitLength >>> 0);
    this.update(padding);
    return Array.from(this.#state, (word) => word.toString(16).padStart(8, '0')).join('');
  }

  // One 64-byte block. Written with locals rather than array destructuring or
  // repeated field access: this loop is the whole cost of hashing a file.
  #compress(bytes, offset) {
    const w = this.#words;
    const s = this.#state;
    for (let i = 0; i < 16; i++) {
      const j = offset + i * 4;
      w[i] = (bytes[j] << 24) | (bytes[j + 1] << 16) | (bytes[j + 2] << 8) | bytes[j + 3];
    }
    for (let i = 16; i < 64; i++) {
      const x = w[i - 15];
      const y = w[i - 2];
      const s0 = rotr(x, 7) ^ rotr(x, 18) ^ (x >>> 3);
      const s1 = rotr(y, 17) ^ rotr(y, 19) ^ (y >>> 10);
      w[i] = (w[i - 16] + s0 + w[i - 7] + s1) | 0;
    }
    let a = s[0], b = s[1], c = s[2], d = s[3], e = s[4], f = s[5], g = s[6], h = s[7];
    for (let i = 0; i < 64; i++) {
      const t1 = (h + (rotr(e, 6) ^ rotr(e, 11) ^ rotr(e, 25)) + ((e & f) ^ (~e & g)) + K[i] + w[i]) | 0;
      const t2 = ((rotr(a, 2) ^ rotr(a, 13) ^ rotr(a, 22)) + ((a & b) ^ (a & c) ^ (b & c))) | 0;
      h = g; g = f; f = e; e = (d + t1) | 0;
      d = c; c = b; b = a; a = (t1 + t2) | 0;
    }
    s[0] += a; s[1] += b; s[2] += c; s[3] += d; s[4] += e; s[5] += f; s[6] += g; s[7] += h;
  }
}
