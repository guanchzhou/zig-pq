//! An inverted-file index with product-quantized residuals (IVF-PQ, Jégou et al. 2011). A coarse
//! k-means quantizer splits the vectors into `nlist` lists; each vector is stored in the list of its
//! nearest coarse centroid as the PQ code of its residual. A query visits the `nprobe` lists whose
//! centroids are nearest and ranks their vectors by the ADC distance of the query's residual,
//! which equals the squared distance to the vector's reconstruction `c + pq(x - c)`.

const std = @import("std");
const kmeans = @import("kmeans.zig");
const pq = @import("pq.zig");

pub const Error = pq.Error || error{TooManyVectors};

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

    /// Adds `data` (row-major, width `d`) with ids `first_id, first_id + 1, ...`. Returns
    /// `error.TooManyVectors`, adding nothing, if the last id would not fit in a `u32`.
    pub fn add(ix: *Index, gpa: std.mem.Allocator, data: []const f32, first_id: u32) Error!void {
        const d = ix.d;
        if (d == 0 or data.len % d != 0) return error.DimensionMismatch;
        const n = data.len / d;
        if (n > 0 and n - 1 > std.math.maxInt(u32) - first_id) return error.TooManyVectors;
        const res = try gpa.alloc(f32, d);
        defer gpa.free(res);
        const code = try gpa.alloc(u8, ix.codebook.m);
        defer gpa.free(code);
        for (0..n) |i| {
            const x = data[i * d ..][0..d];
            const l = kmeans.nearest(ix.coarse, x).index;
            for (res, x, ix.coarse[l * d ..][0..d]) |*r, a, c| r.* = a - c;
            try ix.codebook.encode(res, code);
            try ix.ids[l].ensureUnusedCapacity(gpa, 1);
            try ix.codes[l].ensureUnusedCapacity(gpa, code.len);
            ix.ids[l].appendAssumeCapacity(first_id + @as(u32, @intCast(i)));
            ix.codes[l].appendSliceAssumeCapacity(code);
        }
    }
};

/// An empty index from given coarse centroids and a residual codebook. Ownership of both moves
/// into the index only on success; on error the caller still owns and must free them.
pub fn fromParts(gpa: std.mem.Allocator, d: usize, coarse: []f32, codebook: pq.Codebook) Error!Index {
    if (d == 0 or coarse.len == 0 or coarse.len % d != 0 or codebook.d != d) return error.InvalidShape;
    const nlist = coarse.len / d;
    if (nlist > std.math.maxInt(u32)) return error.InvalidShape;
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
    if (d == 0 or nlist == 0 or data.len % d != 0) return error.InvalidShape;
    const n = data.len / d;
    const coarse = try gpa.alloc(f32, std.math.mul(usize, nlist, d) catch return error.InvalidShape);
    var ix = ix: {
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
        errdefer cb.deinit(gpa);
        break :ix try fromParts(gpa, d, coarse, cb);
    };
    errdefer ix.deinit(gpa);
    try ix.add(gpa, data, 0);
    return ix;
}

pub const Hit = struct { id: u32, dist2: f32 };

fn hitBefore(_: void, a: Hit, b: Hit) bool {
    return a.dist2 < b.dist2 or (a.dist2 == b.dist2 and a.id < b.id);
}

fn siftUpMax(heap: []Hit, idx_in: usize) void {
    var i = idx_in;
    while (i > 0) {
        const parent = (i - 1) / 2;
        if (!hitBefore({}, heap[parent], heap[i])) break;
        std.mem.swap(Hit, &heap[parent], &heap[i]);
        i = parent;
    }
}

fn siftDownMax(heap: []Hit, idx_in: usize) void {
    var i = idx_in;
    const n = heap.len;
    while (true) {
        var largest = i;
        const left = 2 * i + 1;
        const right = 2 * i + 2;
        if (left < n and hitBefore({}, heap[largest], heap[left])) largest = left;
        if (right < n and hitBefore({}, heap[largest], heap[right])) largest = right;
        if (largest == i) break;
        std.mem.swap(Hit, &heap[i], &heap[largest]);
        i = largest;
    }
}

/// The `out.len` nearest stored vectors by ADC distance among the `nprobe` nearest lists.
pub fn search(gpa: std.mem.Allocator, ix: Index, q: []const f32, nprobe: usize, out: []Hit) Error!usize {
    const d = ix.d;
    if (q.len != d) return error.DimensionMismatch;
    if (out.len == 0 or nprobe == 0 or ix.nlist == 0) return 0;
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
    const m = ix.codebook.m;
    var count: usize = 0;
    for (probes[0..@min(nprobe, ix.nlist)]) |p| {
        for (res, q, ix.coarse[p.l * d ..][0..d]) |*r, a, c| r.* = a - c;
        try ix.codebook.table(res, tab);
        for (ix.ids[p.l].items, 0..) |id, i| {
            const cand: Hit = .{ .id = id, .dist2 = try ix.codebook.adc(tab, ix.codes[p.l].items[i * m ..][0..m]) };
            if (count < out.len) {
                out[count] = cand;
                siftUpMax(out[0 .. count + 1], count);
                count += 1;
            } else if (hitBefore({}, cand, out[0])) {
                out[0] = cand;
                siftDownMax(out[0..count], 0);
            }
        }
    }
    std.mem.sort(Hit, out[0..count], {}, hitBefore);
    return count;
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

test "build and fromParts reject zero dimension and mismatched shapes" {
    const gpa = std.testing.allocator;
    try std.testing.expectError(error.InvalidShape, build(gpa, &.{ 1, 2 }, 0, 1, 1, 1, 1, 0));
    try std.testing.expectError(error.InvalidShape, build(gpa, &.{ 1, 2, 3 }, 2, 1, 1, 1, 1, 0));
    try std.testing.expectError(error.InvalidShape, build(gpa, &.{ 1, 2 }, 2, 0, 1, 1, 1, 0));
}

fn buildAlloc(gpa: std.mem.Allocator, data: []const f32) !void {
    var ix = try build(gpa, data, 4, 3, 2, 1, 2, 11);
    defer ix.deinit(gpa);
    var out: [3]Hit = undefined;
    _ = try search(gpa, ix, data[0..4], 2, &out);
}

test "build, add and search free everything exactly once on allocation failure" {
    var data: [12 * 4]f32 = undefined;
    for (&data, 0..) |*x, i| x.* = @floatFromInt((i * 7) % 11);
    try std.testing.checkAllAllocationFailures(std.testing.allocator, buildAlloc, .{@as([]const f32, &data)});
}

fn fromPartsAlloc(gpa: std.mem.Allocator) !void {
    const cb = try pq.fromCentroids(gpa, 2, 1, 1, &.{ 0, 0, 1, 1 });
    var cb_owned = true;
    defer if (cb_owned) cb.deinit(gpa);
    const coarse = try gpa.dupe(f32, &.{ 0, 0, 5, 5 });
    var coarse_owned = true;
    defer if (coarse_owned) gpa.free(coarse);
    var ix = try fromParts(gpa, 2, coarse, cb);
    cb_owned = false;
    coarse_owned = false;
    defer ix.deinit(gpa);
    try ix.add(gpa, &.{ 0, 0, 5, 5, 1, 1, 6, 6 }, 0);
}

test "fromParts leaves ownership with the caller on error" {
    try std.testing.checkAllAllocationFailures(std.testing.allocator, fromPartsAlloc, .{});
}

test "add rejects ids past maxInt(u32) and keeps lists in step" {
    const gpa = std.testing.allocator;
    const cb = try pq.fromCentroids(gpa, 2, 1, 1, &.{ 0, 0, 1, 1 });
    var ix = try fromParts(gpa, 2, try gpa.dupe(f32, &.{ 0, 0 }), cb);
    defer ix.deinit(gpa);
    const data = [_]f32{ 0, 0, 1, 1, 2, 2 };
    try std.testing.expectError(error.TooManyVectors, ix.add(gpa, &data, std.math.maxInt(u32) - 1));
    try std.testing.expectEqual(@as(usize, 0), ix.ids[0].items.len);
    try ix.add(gpa, &data, std.math.maxInt(u32) - 2);
    try ix.add(gpa, &.{}, std.math.maxInt(u32));
    try std.testing.expectEqual(@as(u32, std.math.maxInt(u32)), ix.ids[0].items[2]);
    try std.testing.expectEqual(ix.ids[0].items.len * cb.m, ix.codes[0].items.len);
}

test "build rejects nlist * d overflow and non-finite data" {
    const gpa = std.testing.allocator;
    try std.testing.expectError(error.InvalidShape, build(gpa, &.{ 1, 2 }, 2, std.math.maxInt(usize), 1, 1, 1, 0));
    try std.testing.expectError(error.NonFiniteValue, build(gpa, &.{ 1, std.math.nan(f32), 3, 4 }, 2, 1, 1, 1, 1, 0));
}
