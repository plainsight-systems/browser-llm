"""Reads a GGUF model file's metadata from its first bytes.

The fixture tools fetch those bytes with an HTTP range request rather than
downloading the model. 16 MiB holds the metadata of every model listed in
web/models.json, vocabulary included.
"""
import struct
import urllib.request

HEADER_BYTES = 16 * 2**20

# gguf_type
U8, I8, U16, I16, U32, I32, F32, BOOL, STRING, ARRAY, U64, I64, F64 = range(13)
SCALAR = {U8: "<B", I8: "<b", U16: "<H", I16: "<h", U32: "<I", I32: "<i",
          F32: "<f", BOOL: "<?", U64: "<Q", I64: "<q", F64: "<d"}


def fetch_header(url):
    request = urllib.request.Request(url, headers={"Range": f"bytes=0-{HEADER_BYTES - 1}"})
    with urllib.request.urlopen(request) as response:
        return response.read()


def read_metadata(data):
    """Returns ({key: value}, offset where the metadata ends). Arrays are lists."""
    pos = 0

    def take(fmt):
        nonlocal pos
        (value,) = struct.unpack_from(fmt, data, pos)
        pos += struct.calcsize(fmt)
        return value

    def take_string():
        nonlocal pos
        length = take("<Q")
        text = data[pos:pos + length].decode("utf-8", errors="replace")
        pos += length
        return text

    def take_value(vtype):
        if vtype == STRING:
            return take_string()
        if vtype == ARRAY:
            etype, count = take("<I"), take("<Q")
            return [take_value(etype) for _ in range(count)]
        return take(SCALAR[vtype])

    assert data[:4] == b"GGUF"
    pos = 8
    take("<Q")  # tensor count
    metadata = {}
    for _ in range(take("<Q")):
        key = take_string()
        metadata[key] = take_value(take("<I"))
    return metadata, pos


def without_tensors(data, metadata_end):
    """The header as a file of metadata alone: the same keys, no tensors.

    llama.cpp refuses a file whose tensors lie past its end, as they do in a
    header; with none declared, it loads the vocabulary.
    """
    return data[:8] + struct.pack("<Q", 0) + data[16:metadata_end]
