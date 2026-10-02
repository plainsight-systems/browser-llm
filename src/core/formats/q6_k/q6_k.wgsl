// Q6_K unpack (format.h). A super-block is 256 weights: 128 bytes of low
// nibbles (ql), 64 of high bit pairs (qh), 16 signed 8-bit scales and an fp16
// d. As ggml's dequantize_row_q6_K decodes it, it is two halves of 128
// weights, each four runs of 32; run r of half h is group 4h + r of the
// block. Weight l of a run takes nibble (r / 2) of ql byte 64h + 32(r % 2) + l,
// bit pair r of qh byte 32h + l, and scale 8h + 2r + l / 16, and is
// (d * scale) * (q - 32), in that order.
//
// On the device a piece holds every block's ql, then every block's qh, then
// every block's scales, then every block's d (device_layout.h); each run's
// ql and qh bytes are four-aligned, so they are read a word at a time.
fn unpack(blocks_in_piece: u32, group: u32) -> array<vec4<f32>, 8> {
    let n = blocks_in_piece;
    let b = group / 8u;
    let h = (group % 8u) / 4u;
    let r = group % 4u;
    let d = unpack2x16float(weights[n * 52u + b / 2u])[b % 2u];
    // The run's two scales, adjacent bytes at an even offset in their word.
    let scale_byte = n * 192u + b * 16u + h * 8u + r * 2u;
    let scales = weights[scale_byte / 4u] >> ((scale_byte % 4u) * 8u);
    let first = d * f32(bitcast<i32>(scales << 24u) >> 24u);
    let second = d * f32(bitcast<i32>((scales >> 8u) << 24u) >> 24u);
    let ql_byte = b * 128u + h * 64u + (r % 2u) * 32u;
    let qh_byte = n * 128u + b * 64u + h * 32u;
    let ql_shift = vec4<u32>((r / 2u) * 4u);
    let qh_shift = vec4<u32>(r * 2u);
    var out: array<vec4<f32>, 8>;
    for (var v = 0u; v < 8u; v++) {
        let ql = weights[(ql_byte + v * 4u) / 4u];
        let qh = weights[(qh_byte + v * 4u) / 4u];
        let lo = (vec4<u32>(ql, ql >> 8u, ql >> 16u, ql >> 24u) >> ql_shift) & vec4<u32>(0xFu);
        let hi = (vec4<u32>(qh, qh >> 8u, qh >> 16u, qh >> 24u) >> qh_shift) & vec4<u32>(3u);
        let q = vec4<f32>(vec4<i32>(lo | (hi << vec4<u32>(4u))) - vec4<i32>(32));
        // Weights 0 .. 15 of the run take its first scale, 16 .. 31 its second.
        out[v] = select(second, first, v < 4u) * q;
    }
    return out;
}
