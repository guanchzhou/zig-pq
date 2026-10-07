//! Product quantization (Jégou, Douze and Schmid 2011). A vector of width `d` is split into `m`
//! subvectors of width `d / m`; each is replaced by the index of its nearest centroid in that
//! subspace's codebook of `2^nbits` centroids. The squared distance from a query to a coded vector
//! is estimated asymmetrically (ADC): the query stays exact, and the distance is the sum, over the
//! subspaces, of a lookup table of query-to-centroid distances.

const std = @import("std");
const kmeans = @import("kmeans.zig");

pub const Error = kmeans.Error || error{ InvalidShape, BitsOutOfRange };

pub const Codebook = struct {
    d: usize,
    m: usize,
    nbits: u8,
    /// `m * ksub * dsub` floats: subspace, then centroid, then component.
    centroids: []f32,

    pub fn ksub(cb: Codebook) usize {
        return @as(usize, 1) << @intCast(cb.nbits);
    }

    pub fn dsub(cb: Codebook) usize {
        return cb.d / cb.m;
    }

    pub fn deinit(cb: Codebook, gpa: std.mem.Allocator) void {
        gpa.free(cb.centroids);
    }

    fn sub(cb: Codebook, j: usize) []const f32 {
        const w = cb.ksub() * cb.dsub();
        return cb.centroids[j * w ..][0..w];
    }

    /// Writes the `m` centroid indices of `x` into `out`.
    pub fn encode(cb: Codebook, x: []const f32, out: []u8) void {
        std.debug.assert(x.len == cb.d and out.len == cb.m);
        const s = cb.dsub();
        for (0..cb.m) |j| out[j] = @intCast(kmeans.nearest(cb.sub(j), x[j * s ..][0..s]).index);
    }

    /// The ADC table for query `q`: `m * ksub` squared distances.
    pub fn table(cb: Codebook, q: []const f32, out: []f32) void {
        std.debug.assert(q.len == cb.d and out.len == cb.m * cb.ksub());
        const s = cb.dsub();
        for (0..cb.m) |j| {
            const c = cb.sub(j);
            for (0..cb.ksub()) |k| out[j * cb.ksub() + k] = @floatCast(kmeans.dist2(c[k * s ..][0..s], q[j * s ..][0..s]));
        }
    }

    pub fn adc(cb: Codebook, tab: []const f32, code: []const u8) f32 {
        var s: f32 = 0;
        for (code, 0..) |c, j| s += tab[j * cb.ksub() + c];
        return s;
    }

    /// Writes the reconstruction of `code`.
    pub fn decode(cb: Codebook, code: []const u8, out: []f32) void {
        const s = cb.dsub();
        for (code, 0..) |c, j| @memcpy(out[j * s ..][0..s], cb.sub(j)[@as(usize, c) * s ..][0..s]);
    }
};

/// Copies given centroids into a codebook.
pub fn fromCentroids(gpa: std.mem.Allocator, d: usize, m: usize, nbits: u8, centroids: []const f32) Error!Codebook {
    if (m == 0 or d % m != 0) return error.InvalidShape;
    if (nbits == 0 or nbits > 8) return error.BitsOutOfRange;
    if (centroids.len != (@as(usize, 1) << @intCast(nbits)) * d) return error.InvalidShape;
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
    cb.encode(data[0..d], &code);
    cb.decode(&code, &rec);
    cb.table(data[d .. 2 * d], &tab);
    // ADC equals the exact distance from the query to the reconstruction.
    try std.testing.expectApproxEqRel(@as(f32, @floatCast(kmeans.dist2(&rec, data[d .. 2 * d]))), cb.adc(&tab, &code), 1e-5);
    // The reconstruction is closer than the zero vector, on average.
    var err: f64 = 0;
    var base: f64 = 0;
    for (0..n) |i| {
        cb.encode(data[i * d ..][0..d], &code);
        cb.decode(&code, &rec);
        err += kmeans.dist2(&rec, data[i * d ..][0..d]);
        for (data[i * d ..][0..d]) |x| base += @as(f64, x) * x;
    }
    try std.testing.expect(err < 0.6 * base);
}
