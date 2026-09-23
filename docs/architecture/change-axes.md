# Change axes

One translation unit, one reason to change. This document names the reasons,
because without that list the rule cannot be applied — every split sounds
arguable and the argument is settled by taste.

It governs how [`logical-overview.md`](logical-overview.md)'s boxes become
files. It does not list the files; that is the mapping, written next.

## The axes

| Axis | Triggered by | May change |
|---|---|---|
| **A** | a new model family | graph, tokenizer algorithm, activation |
| **B** | a new quantization type | unpack |
| **C** | a GGUF format revision, or our parse contract | reader, its types, its errors |
| **D** | the WebGPU surface, or granted device limits | device, planner, buffer writer |
| **E** | optimizing one kernel | that kernel only |
| **F** | sampling policy | sampler |
| **G** | cache or context policy | cache, prefix diff |
| **H** | product and interface | the JavaScript presentation layer |
| **I** | **what we claim to support** | one capability table |

Every file maps to exactly one row. Two rows means it splits. Two files that
always change together means they merge — unless they cannot, for a reason
recorded below.

Most of these were found by reading two real model files rather than by
reasoning about the design, which is why the list is short and concrete. See
the provenance table in the logical overview.

## E is a family, not an axis

Optimizing the matmul is not the same reason as optimizing the norm. They move
independently, so the axis is parameterized: E(matmul), E(norm), E(rope),
E(attn), E(gather), E(act). One file each.

The test is mechanical: **a matmul optimization must produce a diff that
touches no other kernel.** If it does, something is shared that should not be.

Three things would recouple them silently:

| Shared thing | If it lives inside kernels | Belongs to |
|---|---|---|
| the unpack function | every kernel changes when a quantization type is added | axis B, its own file |
| bind-group and parameter convention | changing argument order touches all of them | one interface file |
| dispatch geometry arithmetic | retuning workgroup counts touches all of them | its own file |

The third is already done correctly: `core/gpu/dispatch_math` is shared, small,
separately tested, and owned by no kernel.

Workgroup size is the opposite case. It belongs **per kernel**, because the
right value differs per kernel; a shared constant would put all of them back on
one axis.

The six are equal in structure and unequal in value. The matmul is 197 of the
~295 dispatches per token and effectively all of the weight bandwidth, so five
of these files are correctness surface and one is the performance surface.

## The boxes that are already one axis

`Gates` (I — it applies the table, it does not contain criteria), `Jinja` (H —
it renders a template it does not interpret), `Graph` (A — one per family), and
each kernel (E).

## The boxes that are compound

| Box | Axes it mixes | Splits into |
|---|---|---|
| **Upload** | D, B, D | planner · unpack · buffer writer |
| **Fetch** | H, G | transport with progress · OPFS cache |
| **Stream** | F, A, H | sampler · detokenize · emit |
| **Diff KV** | G, D | prefix diff · cache resources |
| **Pick** | H, and data | picker interface · a curated list that is JSON, not code |

**Diff KV** is the split worth insisting on. `longest_common_prefix(old, new)
→ length` is a pure function over two integer sequences: no GPU, no model, no
file, no browser. The cache is buffers and counters. Fused, the most
correctness-critical logic in the chat loop becomes reachable only through a
device.

**Upload** is the one that carries the most risk if left whole. The planner is
pure arithmetic over a tensor index and a limit set, and preflight depends on
running it before any weight byte is fetched. Fusing it with the writer would
make the compatibility gate impossible.

## Two cases that look like violations and are not

**The capability table has one owner and several readers.** The picker's gates,
the unpack dispatch, and the graph selector all consult it. That is not a
violation: the table changes for exactly one reason, and its readers do not
change when it does. Adding Q4_1 means one new unpack file and one new row.

This is also what keeps the picker honest. If the gates and the loader consult
two lists, the lists will diverge, and a picker reporting "compatible" for a
model that then fails to load is worse than no picker. The single table is the
enforcement; review is not.

**A kernel is one responsibility expressed as two files.** The WGSL and its C++
launcher change together — retile the shader and the dispatch geometry moves
with it. The strict rule says merge them; they are different languages, so the
coupling is a language artifact rather than a design choice. They stay a
co-located pair, and nothing should be introduced between them to make the
separation look intentional.

## Where the code violates this today

`bytes_for_elements` in `core/gguf/reader.cpp` switches on `TensorType` and
returns `false` for anything outside {F32, F16, Q4_0}. That `default` branch is
the capability table, living inside the GGUF reader.

So adding Q4_1 edits the reader: a file on axis C changing for an axis I
reason. It also reaches into `quant::` for block constants, which makes the
container parser depend on quantization details it has no business knowing.

`core/gguf/types.h` is *not* a violation, which is worth recording because it
looks like one. Magic, version, limits, `ValueType`, `TensorType`, `ReadError`,
and the parse output types are all descriptions of the GGUF format. They change
for one reason. `TensorType` enumerating quantized formats is format
knowledge — *which* of them we implement is the capability table's business,
and lives elsewhere.

## How to apply this

Before adding to a file, ask which row of the table would cause this line to
change. If it is not the row that file already owns, it belongs somewhere else.

Before creating a file, name its row. A file whose row cannot be named is
either a utility bucket or a responsibility nobody has articulated, and both
tend to accumulate.

The decisive question for a split is not "are these different concerns" —
that is unfalsifiable — but "can I name a change that touches one and not the
other." If not, they are one file.
