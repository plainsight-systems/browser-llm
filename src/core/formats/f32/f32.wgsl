// F32 unpack (format.h). A "block" is one float, stored whole, so group g of
// a piece is its floats 32g .. 32g + 31, read as words and reinterpreted:
// nothing is computed, so every value, -0 and subnormals included, comes back
// as stored. The caller asks only for groups within the piece.
fn unpack(blocks_in_piece: u32, group: u32) -> array<vec4<f32>, 8> {
    var out: array<vec4<f32>, 8>;
    for (var v = 0u; v < 8u; v++) {
        let at = group * 32u + v * 4u;
        out[v] = bitcast<vec4<f32>>(vec4<u32>(weights[at], weights[at + 1u], weights[at + 2u], weights[at + 3u]));
    }
    return out;
}
