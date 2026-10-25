//! Fuzz targets checking invariants that must hold for every input. `zig build test` runs every
//! input in `src/fuzz-corpus/`; `zig build test --fuzz` explores further.

const std = @import("std");
const kmeans = @import("kmeans.zig");
const pq = @import("pq.zig");
const ivf = @import("ivf.zig");
const Smith = std.testing.Smith;

const corpus = [_][]const u8{
    @embedFile("fuzz-corpus/00"),
    @embedFile("fuzz-corpus/01"),
    @embedFile("fuzz-corpus/02"),
    @embedFile("fuzz-corpus/03"),
};

fn upTo(smith: *Smith, lo: usize, hi: usize) usize {
    return smith.valueRangeAtMost(u32, @intCast(lo), @intCast(hi));
}

fn invariants(_: void, smith: *Smith) anyerror!void {
    const gpa = std.testing.allocator;
    const m = upTo(smith, 1, 4);
    const d = m * upTo(smith, 1, 4);
    const nbits: u8 = @intCast(upTo(smith, 1, 4));
    const n = upTo(smith, (@as(usize, 1) << @intCast(nbits)) + 1, 80);
    var data: [80 * 16]f32 = undefined;
    for (data[0 .. n * d]) |*x| x.* = @as(f32, @floatFromInt(smith.value(i16))) / 2048;
    const x = data[0 .. n * d];

    // Lloyd never increases the objective.
    var cents: [16 * 16]f32 = undefined;
    const k = @as(usize, 1) << @intCast(nbits);
    _ = try kmeans.train(gpa, x, d, k, 0, smith.value(u64), cents[0 .. k * d]);
    var assign: [80]u32 = undefined;
    var prev = std.math.inf(f64);
    for (0..4) |_| {
        const o = try kmeans.lloydStep(gpa, x, d, cents[0 .. k * d], assign[0..n]);
        try std.testing.expect(o <= prev * (1 + 1e-9) + 1e-9);
        prev = o;
    }

    // ADC equals the exact distance to the reconstruction.
    const cb = try pq.train(gpa, x, d, m, nbits, 3, smith.value(u64));
    defer cb.deinit(gpa);
    var code: [4]u8 = undefined;
    var rec: [16]f32 = undefined;
    var tab: [4 * 16]f32 = undefined;
    try cb.table(x[0..d], tab[0 .. m * k]);
    for (0..n) |i| {
        try cb.encode(x[i * d ..][0..d], code[0..m]);
        for (code[0..m]) |c| try std.testing.expect(c < k);
        try cb.decode(code[0..m], rec[0..d]);
        const exact: f32 = @floatCast(kmeans.dist2(rec[0..d], x[0..d]));
        try std.testing.expect(@abs(try cb.adc(tab[0 .. m * k], code[0..m]) - exact) <= 1e-4 * @max(1, exact));
    }

    // IVF stores each vector once.
    var ix = try ivf.build(gpa, x, d, upTo(smith, 1, @min(8, n)), m, nbits, 2, smith.value(u64));
    defer ix.deinit(gpa);
    var seen: [80]bool = @splat(false);
    for (ix.ids) |l| for (l.items) |id| {
        try std.testing.expect(!seen[id]);
        seen[id] = true;
    };
    for (seen[0..n]) |s| try std.testing.expect(s);
}

test "fuzz: k-means, codes, ADC and IVF stay consistent" {
    try std.testing.fuzz({}, invariants, .{ .corpus = &corpus });
}
