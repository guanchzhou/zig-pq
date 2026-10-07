//! An inverted-file index with product-quantized residuals (IVF-PQ, Jégou et al. 2011). A coarse
//! k-means quantizer splits the vectors into `nlist` lists; each vector is stored in the list of its
//! nearest coarse centroid as the PQ code of its residual. A query visits the `nprobe` lists whose
//! centroids are nearest and ranks their vectors by the ADC distance of the query's residual,
//! which equals the squared distance to the vector's reconstruction `c + pq(x - c)`.

const std = @import("std");
const kmeans = @import("kmeans.zig");
const pq = @import("pq.zig");

pub const Error = pq.Error;

pub const Index = struct {
    d: usize,
    nlist: usize,
    coarse: []f32,
    codebook: pq.Codebook,
    /// Per list: vector ids and their codes (`m` bytes each), in insertion order.
    ids: []std.ArrayList(u32),
    codes: []std.ArrayList(u8),

    pub fn deinit(ix: *Index, gpa: std.mem.Allocator) void {
        gpa.free(ix.coarse);
        ix.codebook.deinit(gpa);
        for (ix.ids, ix.codes) |*a, *b| {
            a.deinit(gpa);
            b.deinit(gpa);
        }
        gpa.free(ix.ids);
        gpa.free(ix.codes);
    }

    /// Adds `data` (row-major, width `d`) with ids `first_id, first_id + 1, ...`.
    pub fn add(ix: *Index, gpa: std.mem.Allocator, data: []const f32, first_id: u32) Error!void {
        const d = ix.d;
        const res = try gpa.alloc(f32, d);
        defer gpa.free(res);
        const code = try gpa.alloc(u8, ix.codebook.m);
        defer gpa.free(code);
        for (0..data.len / d) |i| {
            const x = data[i * d ..][0..d];
            const l = kmeans.nearest(ix.coarse, x).index;
            for (res, x, ix.coarse[l * d ..][0..d]) |*r, a, c| r.* = a - c;
            ix.codebook.encode(res, code);
            try ix.ids[l].append(gpa, first_id + @as(u32, @intCast(i)));
            try ix.codes[l].appendSlice(gpa, code);
        }
    }
};

/// An empty index from given coarse centroids and a residual codebook (ownership moves in).
pub fn fromParts(gpa: std.mem.Allocator, d: usize, coarse: []f32, codebook: pq.Codebook) Error!Index {
    const nlist = coarse.len / d;
    const ids = try gpa.alloc(std.ArrayList(u32), nlist);
    errdefer gpa.free(ids);
    const codes = try gpa.alloc(std.ArrayList(u8), nlist);
    for (ids, codes) |*a, *b| {
        a.* = .empty;
        b.* = .empty;
    }
    return .{ .d = d, .nlist = nlist, .coarse = coarse, .codebook = codebook, .ids = ids, .codes = codes };
}

/// Trains the coarse quantizer and the residual codebook on `data`, then adds it.
pub fn build(gpa: std.mem.Allocator, data: []const f32, d: usize, nlist: usize, m: usize, nbits: u8, iters: usize, seed: u64) Error!Index {
    const n = data.len / d;
    const coarse = try gpa.alloc(f32, nlist * d);
    errdefer gpa.free(coarse);
    _ = try kmeans.train(gpa, data, d, nlist, iters, seed, coarse);
    const res = try gpa.alloc(f32, data.len);
    defer gpa.free(res);
    for (0..n) |i| {
        const x = data[i * d ..][0..d];
        const l = kmeans.nearest(coarse, x).index;
        for (res[i * d ..][0..d], x, coarse[l * d ..][0..d]) |*r, a, c| r.* = a - c;
    }
    const cb = try pq.train(gpa, res, d, m, nbits, iters, kmeans.splitmix64(seed, 0xc0de));
    var ix = try fromParts(gpa, d, coarse, cb);
    errdefer ix.deinit(gpa);
    try ix.add(gpa, data, 0);
    return ix;
}

pub const Hit = struct { id: u32, dist2: f32 };

fn hitBefore(_: void, a: Hit, b: Hit) bool {
    return a.dist2 < b.dist2 or (a.dist2 == b.dist2 and a.id < b.id);
}

/// The `out.len` nearest stored vectors by ADC distance among the `nprobe` nearest lists.
pub fn search(gpa: std.mem.Allocator, ix: Index, q: []const f32, nprobe: usize, out: []Hit) Error!usize {
    const d = ix.d;
    const Probe = struct { l: u32, v: f64 };
    const probes = try gpa.alloc(Probe, ix.nlist);
    defer gpa.free(probes);
    for (probes, 0..) |*p, l| p.* = .{ .l = @intCast(l), .v = kmeans.dist2(ix.coarse[l * d ..][0..d], q) };
    std.mem.sort(Probe, probes, {}, struct {
        fn f(_: void, a: Probe, b: Probe) bool {
            return a.v < b.v or (a.v == b.v and a.l < b.l);
        }
    }.f);
    const tab = try gpa.alloc(f32, ix.codebook.m * ix.codebook.ksub());
    defer gpa.free(tab);
    const res = try gpa.alloc(f32, d);
    defer gpa.free(res);
    var hits: std.ArrayList(Hit) = .empty;
    defer hits.deinit(gpa);
    const m = ix.codebook.m;
    for (probes[0..@min(nprobe, ix.nlist)]) |p| {
        for (res, q, ix.coarse[p.l * d ..][0..d]) |*r, a, c| r.* = a - c;
        ix.codebook.table(res, tab);
        for (ix.ids[p.l].items, 0..) |id, i| {
            try hits.append(gpa, .{ .id = id, .dist2 = ix.codebook.adc(tab, ix.codes[p.l].items[i * m ..][0..m]) });
        }
    }
    std.mem.sort(Hit, hits.items, {}, hitBefore);
    const k = @min(out.len, hits.items.len);
    @memcpy(out[0..k], hits.items[0..k]);
    return k;
}

test "every vector finds itself with all lists probed" {
    const gpa = std.testing.allocator;
    var prng = std.Random.DefaultPrng.init(6);
    const r = prng.random();
    const n = 600;
    const d = 16;
    var data: [n * d]f32 = undefined;
    for (&data) |*x| x.* = r.floatNorm(f32);
    var ix = try build(gpa, &data, d, 8, 8, 5, 8, 3);
    defer ix.deinit(gpa);
    var total: usize = 0;
    for (ix.ids) |l| total += l.items.len;
    try std.testing.expectEqual(@as(usize, n), total);
    var found: usize = 0;
    var out: [10]Hit = undefined;
    for (0..50) |i| {
        const k = try search(gpa, ix, data[i * d ..][0..d], 8, &out);
        for (out[0..k]) |h| found += @intFromBool(h.id == i);
    }
    try std.testing.expect(found >= 45);
}
