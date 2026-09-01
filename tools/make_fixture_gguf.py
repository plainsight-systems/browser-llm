#!/usr/bin/env python3
"""Builds small synthetic GGUF files for the reader tests.

Fixtures are generated rather than committed as opaque blobs so a reviewer can
see exactly what each byte is, and so a malformed case is one deliberate
mutation away from the valid one rather than a hand-edited binary nobody can
audit.

CI must never download the real 420 MB model, so every automated test runs
against these.
"""
import struct
import sys

MAGIC = b"GGUF"
VERSION = 3

# gguf_type
U8, I8, U16, I16, U32, I32, F32, BOOL, STRING, ARRAY, U64, I64, F64 = range(13)

# ggml_type
T_F32, T_F16, T_Q4_0 = 0, 1, 2


def gstr(s: bytes) -> bytes:
    return struct.pack("<Q", len(s)) + s


def kv(key: bytes, vtype: int, payload: bytes) -> bytes:
    return gstr(key) + struct.pack("<I", vtype) + payload


def build(tensors, metadata=(), *, magic=MAGIC, version=VERSION,
          tensor_count=None, metadata_count=None, alignment=32):
    """tensors: list of (name, dims, ggml_type, data_bytes)."""
    md = b"".join(metadata)
    md_count = metadata_count if metadata_count is not None else len(metadata)
    t_count = tensor_count if tensor_count is not None else len(tensors)

    head = magic + struct.pack("<I", version)
    head += struct.pack("<QQ", t_count, md_count)
    head += md

    # Lay tensors out back to back, each padded to `alignment`.
    infos, blobs, offset = b"", b"", 0
    for name, dims, ttype, data in tensors:
        infos += gstr(name) + struct.pack("<I", len(dims))
        for d in dims:
            infos += struct.pack("<q", d)
        infos += struct.pack("<I", ttype) + struct.pack("<Q", offset)
        pad = (-len(data)) % alignment
        blobs += data + b"\0" * pad
        offset += len(data) + pad

    body = head + infos
    body += b"\0" * ((-len(body)) % alignment)
    return body + blobs


def q4_0_blocks(n: int, scale_bits: int = 0x3C00, nibble_byte: int = 0x10) -> bytes:
    """n Q4_0 blocks: fp16 scale then 16 bytes of packed nibbles."""
    return (struct.pack("<H", scale_bits) + bytes([nibble_byte]) * 16) * n


def valid() -> bytes:
    return build(
        tensors=[
            (b"token_embd.weight", [64, 2], T_Q4_0, q4_0_blocks(4)),
            (b"output_norm.weight", [4], T_F32, struct.pack("<4f", 1.0, 2.0, 3.0, 4.0)),
        ],
        metadata=[
            kv(b"general.architecture", STRING, gstr(b"qwen3")),
            kv(b"general.alignment", U32, struct.pack("<I", 32)),
            kv(b"qwen3.block_count", U32, struct.pack("<I", 28)),
            kv(b"qwen3.attention.head_count", U32, struct.pack("<I", 16)),
            kv(b"tokenizer.ggml.tokens", ARRAY,
               struct.pack("<IQ", STRING, 3) + gstr(b"a") + gstr(b"bb") + gstr(b"ccc")),
        ],
    )


def _with_tensor_offset(bogus_offset: int) -> bytes:
    """A structurally valid file whose single tensor claims an absurd offset."""
    data = q4_0_blocks(1)
    head = MAGIC + struct.pack("<I", VERSION) + struct.pack("<QQ", 1, 0)
    info = gstr(b"t") + struct.pack("<I", 1) + struct.pack("<q", 32)
    info += struct.pack("<I", T_Q4_0) + struct.pack("<Q", bogus_offset)
    body = head + info
    body += b"\0" * ((-len(body)) % 32)
    return body + data + b"\0" * ((-len(data)) % 32)


CASES = {
    "valid": valid,
    "bad_magic": lambda: build([], magic=b"GGUX"),
    "bad_version": lambda: build([], version=2),
    "truncated_header": lambda: MAGIC + struct.pack("<I", VERSION) + b"\x01\x02",
    # Claims more tensors than the file can possibly contain.
    "tensor_count_too_large": lambda: build([], tensor_count=1 << 40),
    "metadata_count_lies": lambda: build([], metadata_count=5),
    # A tensor whose DECLARED offset puts its data past the end of the file.
    # Trimming trailing padding would not do this: the padding is not part of
    # any declared region, so removing it leaves every tensor still resident.
    "offset_past_eof": lambda: _with_tensor_offset(1 << 40),
    # An offset chosen so data_start + offset WRAPS past 2^64 rather than
    # merely exceeding the file. A naive `start + offset > size` test computes
    # a small number here and reports the region as in-bounds.
    "offset_overflow": lambda: _with_tensor_offset((1 << 64) - 64),
    # Truncated so the cut lands inside declared tensor data, not padding.
    "data_truncated": lambda: build(
        [(b"t", [64], T_Q4_0, q4_0_blocks(2))]
    )[:-40],
    "negative_dimension": lambda: build(
        [(b"t", [-4], T_F32, b"\0" * 16)]
    ),
    "unknown_tensor_type": lambda: build(
        [(b"t", [4], 999, b"\0" * 16)]
    ),
    "not_block_aligned": lambda: build(
        # 33 elements is not a whole number of Q4_0 blocks.
        [(b"t", [33], T_Q4_0, q4_0_blocks(2))]
    ),
    "duplicate_tensor_name": lambda: build([
        (b"dup", [4], T_F32, b"\0" * 16),
        (b"dup", [4], T_F32, b"\0" * 16),
    ]),
    "nested_array": lambda: build(
        [], metadata=[kv(b"bad", ARRAY, struct.pack("<IQ", ARRAY, 1))]
    ),
    "unknown_value_type": lambda: build(
        [], metadata=[kv(b"bad", 99, b"\0" * 4)]
    ),
}


def main() -> int:
    if len(sys.argv) != 3 or sys.argv[1] not in CASES:
        print(f"usage: {sys.argv[0]} <case> <out>\ncases: {', '.join(CASES)}",
              file=sys.stderr)
        return 2
    with open(sys.argv[2], "wb") as f:
        f.write(CASES[sys.argv[1]]())
    return 0


if __name__ == "__main__":
    sys.exit(main())
