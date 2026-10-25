//! Product quantization (Jégou, Douze and Schmid 2011). A vector of width `d` is split into `m`
//! subvectors of width `d / m`; each is replaced by the index of its nearest centroid in that
//! subspace's codebook of `2^nbits` centroids. The squared distance from a query to a coded vector
//! is estimated asymmetrically (ADC): the query stays exact, and the distance is the sum, over the
//! subspaces, of a lookup table of query-to-centroid distances.

const std = @import("std");
const kmeans = @import("kmeans.zig");

pub const Error = kmeans.Error || error{ InvalidShape, BitsOutOfRange, InvalidCode };

pub const Codebook = struct {
    d: usize,
    m: usize,
    nbits: u8,
    /// `m * ksub * dsub` floats: subspace, then centroid, then component.
    centroids: []f32,

    /// `2^nbits`, or 0 when `nbits` is too large to shift by.
    pub fn ksub(cb: Codebook) usize {
        return std.math.shl(usize, 1, cb.nbits);
    }

    pub fn dsub(cb: Codebook) usize {
        return if (cb.m == 0) 0 else cb.d / cb.m;
    }

    pub fn deinit(cb: Codebook, gpa: std.mem.Allocator) void {
        gpa.free(cb.centroids);
    }

    fn validate(cb: Codebook) Error!void {
        if (cb.nbits == 0 or cb.nbits > 8) return error.BitsOutOfRange;
        if (cb.m == 0 or cb.d == 0 or cb.d % cb.m != 0 or cb.centroids.len != cb.ksub() * cb.d) return error.InvalidShape;
    }

    fn sub(cb: Codebook, j: usize) []const f32 {
        const w = cb.ksub() * cb.dsub();
        return cb.centroids[j * w ..][0..w];
    }

    fn checkCode(cb: Codebook, code: []const u8) Error!void {
        if (code.len != cb.m) return error.DimensionMismatch;
        const k = cb.ksub();
        for (code) |c| if (c >= k) return error.InvalidCode;
    }

    /// Writes the `m` centroid indices of `x` into `out`.
    pub fn encode(cb: Codebook, x: []const f32, out: []u8) Error!void {
        try cb.validate();
        if (x.len != cb.d or out.len != cb.m) return error.DimensionMismatch;
        const s = cb.dsub();
        for (0..cb.m) |j| out[j] = @intCast(kmeans.nearest(cb.sub(j), x[j * s ..][0..s]).index);
    }

    /// The ADC table for query `q`: `m * ksub` squared distances.
    pub fn table(cb: Codebook, q: []const f32, out: []f32) Error!void {
        try cb.validate();
        if (q.len != cb.d or out.len != cb.m * cb.ksub()) return error.DimensionMismatch;
        const s = cb.dsub();
        for (0..cb.m) |j| {
            const c = cb.sub(j);
            for (0..cb.ksub()) |k| out[j * cb.ksub() + k] = @floatCast(kmeans.dist2(c[k * s ..][0..s], q[j * s ..][0..s]));
        }
    }

    /// The ADC distance of `code` from a table written by `table`.
    pub fn adc(cb: Codebook, tab: []const f32, code: []const u8) Error!f32 {
        try cb.validate();
        if (tab.len != cb.m * cb.ksub()) return error.DimensionMismatch;
        try cb.checkCode(code);
        var s: f32 = 0;
        for (code, 0..) |c, j| s += tab[j * cb.ksub() + c];
        return s;
    }

    /// Writes the reconstruction of `code`.
    pub fn decode(cb: Codebook, code: []const u8, out: []f32) Error!void {
        try cb.validate();
        if (out.len != cb.d) return error.DimensionMismatch;
        try cb.checkCode(code);
        const s = cb.dsub();
        for (code, 0..) |c, j| @memcpy(out[j * s ..][0..s], cb.sub(j)[@as(usize, c) * s ..][0..s]);
    }
};

/// Copies given centroids into a codebook.
pub fn fromCentroids(gpa: std.mem.Allocator, d: usize, m: usize, nbits: u8, centroids: []const f32) Error!Codebook {
    if (m == 0 or d == 0 or d % m != 0) return error.InvalidShape;
    if (nbits == 0 or nbits > 8) return error.BitsOutOfRange;
    if (centroids.len != std.math.mul(usize, @as(usize, 1) << @intCast(nbits), d) catch return error.InvalidShape) return error.InvalidShape;
    return .{ .d = d, .m = m, .nbits = nbits, .centroids = try gpa.dupe(f32, centroids) };
}

/// Trains one codebook per subspace with k-means (subspace `j` seeded with `splitmix64(seed, j)`).
pub fn train(gpa: std.mem.Allocator, data: []const f32, d: usize, m: usize, nbits: u8, iters: usize, seed: u64) Error!Codebook {
    if (m == 0 or d == 0 or d % m != 0 or data.len % d != 0) return error.InvalidShape;
    if (nbits == 0 or nbits > 8) return error.BitsOutOfRange;
    const n = data.len / d;
    const s = d / m;
    const k = @as(usize, 1) << @intCast(nbits);
    const cents = try gpa.alloc(f32, m * k * s);
    errdefer gpa.free(cents);
    const subdata = try gpa.alloc(f32, n * s);
    defer gpa.free(subdata);
    for (0..m) |j| {
        for (0..n) |i| @memcpy(subdata[i * s ..][0..s], data[i * d + j * s ..][0..s]);
        _ = try kmeans.train(gpa, subdata, s, k, iters, kmeans.splitmix64(seed, j), cents[j * k * s ..][0 .. k * s]);
    }
    return .{ .d = d, .m = m, .nbits = nbits, .centroids = cents };
}

test "encode, decode and ADC agree" {
    const gpa = std.testing.allocator;
    var prng = std.Random.DefaultPrng.init(2);
    const r = prng.random();
    const n = 400;
    const d = 16;
    var data: [n * d]f32 = undefined;
    for (&data) |*x| x.* = r.floatNorm(f32);
    const cb = try train(gpa, &data, d, 4, 4, 10, 1);
    defer cb.deinit(gpa);
    var code: [4]u8 = undefined;
    var rec: [d]f32 = undefined;
    var tab: [4 * 16]f32 = undefined;
    try cb.encode(data[0..d], &code);
    try cb.decode(&code, &rec);
    try cb.table(data[d .. 2 * d], &tab);
    // ADC equals the exact distance from the query to the reconstruction.
    try std.testing.expectApproxEqRel(@as(f32, @floatCast(kmeans.dist2(&rec, data[d .. 2 * d]))), try cb.adc(&tab, &code), 1e-5);
    // The reconstruction is closer than the zero vector, on average.
    var err: f64 = 0;
    var base: f64 = 0;
    for (0..n) |i| {
        try cb.encode(data[i * d ..][0..d], &code);
        try cb.decode(&code, &rec);
        err += kmeans.dist2(&rec, data[i * d ..][0..d]);
        for (data[i * d ..][0..d]) |x| base += @as(f64, x) * x;
    }
    try std.testing.expect(err < 0.6 * base);
}

test "fromCentroids rejects zero width" {
    try std.testing.expectError(error.InvalidShape, fromCentroids(std.testing.allocator, 0, 1, 1, &.{}));
    try std.testing.expectError(error.InvalidShape, fromCentroids(std.testing.allocator, 0, 3, 2, &.{}));
}

test "encode, table, adc and decode reject bad lengths and codes" {
    const gpa = std.testing.allocator;
    // d = 2, m = 2, nbits = 1: two centroids per one-wide subspace.
    const cb = try fromCentroids(gpa, 2, 2, 1, &.{ 0, 1, 2, 3 });
    defer cb.deinit(gpa);
    var code: [2]u8 = undefined;
    var tab: [4]f32 = undefined;
    var rec: [2]f32 = undefined;
    try std.testing.expectError(error.DimensionMismatch, cb.encode(&.{1}, &code));
    try std.testing.expectError(error.DimensionMismatch, cb.encode(&.{ 1, 2 }, code[0..1]));
    try std.testing.expectError(error.DimensionMismatch, cb.table(&.{ 1, 2 }, tab[0..3]));
    try cb.table(&.{ 1, 2 }, &tab);
    try std.testing.expectError(error.DimensionMismatch, cb.adc(tab[0..2], &.{ 0, 0 }));
    try std.testing.expectError(error.DimensionMismatch, cb.adc(&tab, &.{0}));
    try std.testing.expectError(error.InvalidCode, cb.adc(&tab, &.{ 0, 2 }));
    try std.testing.expectError(error.InvalidCode, cb.adc(&tab, &.{ 255, 0 }));
    try std.testing.expectError(error.InvalidCode, cb.decode(&.{ 0, 2 }, &rec));
    try std.testing.expectError(error.DimensionMismatch, cb.decode(&.{ 0, 1 }, rec[0..1]));
    try cb.decode(&.{ 1, 0 }, &rec);
    try std.testing.expectEqualSlices(f32, &.{ 1, 2 }, &rec);
}

test "a hand-built codebook with an invalid shape is rejected" {
    var cents = [_]f32{ 0, 1 };
    var code: [1]u8 = undefined;
    const zero_m: Codebook = .{ .d = 2, .m = 0, .nbits = 1, .centroids = &cents };
    try std.testing.expectError(error.InvalidShape, zero_m.encode(&.{ 0, 0 }, &code));
    const wide: Codebook = .{ .d = 1, .m = 1, .nbits = 200, .centroids = &cents };
    try std.testing.expectError(error.BitsOutOfRange, wide.encode(&.{0}, &code));
    const short: Codebook = .{ .d = 2, .m = 1, .nbits = 1, .centroids = &cents };
    try std.testing.expectError(error.InvalidShape, short.encode(&.{ 0, 0 }, &code));
}

fn trainAlloc(gpa: std.mem.Allocator, data: []const f32) !void {
    const cb = try train(gpa, data, 4, 2, 1, 3, 5);
    defer cb.deinit(gpa);
    const copy = try fromCentroids(gpa, cb.d, cb.m, cb.nbits, cb.centroids);
    copy.deinit(gpa);
}

test "train and fromCentroids free everything on allocation failure" {
    var data: [8 * 4]f32 = undefined;
    for (&data, 0..) |*x, i| x.* = @floatFromInt(i % 7);
    try std.testing.checkAllAllocationFailures(std.testing.allocator, trainAlloc, .{@as([]const f32, &data)});
}
