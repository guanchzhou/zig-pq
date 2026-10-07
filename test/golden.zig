//! Replays test/golden.json (test/reference.py): encoding and search against Faiss with Faiss's own
//! codebooks, and one Lloyd step against NumPy. Faiss computes in float32 and this library
//! accumulates in float64, so a code may differ only where two centroids are within 1e-5 of each
//! other in distance, and search results may differ only in order among distances within 1e-4.

const std = @import("std");
const zpq = @import("pq");
const kmeans = zpq.kmeans;
const pq = zpq.pq;
const ivf = zpq.ivf;

const Golden = struct {
    pq: []const struct { d: usize, m: usize, nbits: u8, data: []const f32, queries: []const f32, centroids: []const f32, codes: []const u8, dist: []const f32, ids: []const i64 },
    ivf: []const struct { d: usize, m: usize, nbits: u8, nlist: usize, nprobe: usize, data: []const f32, queries: []const f32, coarse: []const f32, centroids: []const f32, dist: []const f32, ids: []const i64 },
    lloyd: []const struct { d: usize, k: usize, data: []const f32, init: []const f32, assign: []const u32, centroids: []const f32, objective: f64 },
};

fn close(want: f32, got: f32, tol: f32) bool {
    return @abs(want - got) <= tol * @max(1, @abs(want));
}

/// Ranked results agree: distances match at every rank, and ids match except among near-equal distances.
fn expectRanked(want_d: []const f32, want_i: []const i64, got: []const ivf.Hit) !void {
    try std.testing.expectEqual(want_d.len, got.len);
    for (want_d, want_i, got, 0..) |wd, wi, g, r| {
        if (!close(wd, g.dist2, 1e-4)) {
            std.debug.print("rank {d}: want {d}, got {d}\n", .{ r, wd, g.dist2 });
            return error.TestExpectedApproxEq;
        }
        if (wi != g.id) {
            var tied = false;
            for (want_d, want_i) |other_d, other_i| tied = tied or (other_i == g.id and close(other_d, g.dist2, 1e-4));
            try std.testing.expect(tied);
        }
    }
}

test "PQ encoding and ADC search match Faiss" {
    var arena_state = std.heap.ArenaAllocator.init(std.testing.allocator);
    defer arena_state.deinit();
    const arena = arena_state.allocator();
    const g = try std.json.parseFromSliceLeaky(Golden, arena, @embedFile("golden.json"), .{ .ignore_unknown_fields = true });
    for (g.pq) |c| {
        const cb = try pq.fromCentroids(arena, c.d, c.m, c.nbits, c.centroids);
        const n = c.data.len / c.d;
        const code = try arena.alloc(u8, c.m);
        var mismatches: usize = 0;
        const s = cb.dsub();
        for (0..n) |i| {
            const x = c.data[i * c.d ..][0..c.d];
            cb.encode(x, code);
            for (code, c.codes[i * c.m ..][0..c.m], 0..) |z, f, j| {
                if (z == f) continue;
                mismatches += 1;
                const sub = cb.centroids[j * cb.ksub() * s ..][0 .. cb.ksub() * s];
                const dz = kmeans.dist2(sub[@as(usize, z) * s ..][0..s], x[j * s ..][0..s]);
                const df = kmeans.dist2(sub[@as(usize, f) * s ..][0..s], x[j * s ..][0..s]);
                try std.testing.expect(@abs(dz - df) <= 1e-5 * @max(1, df));
            }
        }
        try std.testing.expect(mismatches * 1000 <= n * c.m);
        const tab = try arena.alloc(f32, c.m * cb.ksub());
        const hits = try arena.alloc(ivf.Hit, n);
        for (0..c.queries.len / c.d) |qi| {
            cb.table(c.queries[qi * c.d ..][0..c.d], tab);
            for (hits, 0..) |*h, i| h.* = .{ .id = @intCast(i), .dist2 = cb.adc(tab, c.codes[i * c.m ..][0..c.m]) };
            std.mem.sort(ivf.Hit, hits, {}, struct {
                fn f(_: void, a: ivf.Hit, b: ivf.Hit) bool {
                    return a.dist2 < b.dist2 or (a.dist2 == b.dist2 and a.id < b.id);
                }
            }.f);
            try expectRanked(c.dist[qi * 10 ..][0..10], c.ids[qi * 10 ..][0..10], hits[0..10]);
        }
    }
}

test "IVF-PQ search matches Faiss" {
    var arena_state = std.heap.ArenaAllocator.init(std.testing.allocator);
    defer arena_state.deinit();
    const arena = arena_state.allocator();
    const g = try std.json.parseFromSliceLeaky(Golden, arena, @embedFile("golden.json"), .{ .ignore_unknown_fields = true });
    for (g.ivf) |c| {
        const cb = try pq.fromCentroids(arena, c.d, c.m, c.nbits, c.centroids);
        var ix = try ivf.fromParts(arena, c.d, try arena.dupe(f32, c.coarse), cb);
        try ix.add(arena, c.data, 0);
        var out: [10]ivf.Hit = undefined;
        for (0..c.queries.len / c.d) |qi| {
            const k = try ivf.search(arena, ix, c.queries[qi * c.d ..][0..c.d], c.nprobe, &out);
            try expectRanked(c.dist[qi * 10 ..][0..10], c.ids[qi * 10 ..][0..10], out[0..k]);
        }
    }
}

test "one Lloyd step matches NumPy" {
    var arena_state = std.heap.ArenaAllocator.init(std.testing.allocator);
    defer arena_state.deinit();
    const arena = arena_state.allocator();
    const g = try std.json.parseFromSliceLeaky(Golden, arena, @embedFile("golden.json"), .{ .ignore_unknown_fields = true });
    for (g.lloyd) |c| {
        const cents = try arena.dupe(f32, c.init);
        const assign = try arena.alloc(u32, c.data.len / c.d);
        const obj = try kmeans.lloydStep(arena, c.data, c.d, cents, assign);
        try std.testing.expectEqualSlices(u32, c.assign, assign);
        for (c.centroids, cents) |w, v| try std.testing.expectApproxEqAbs(w, v, 1e-5);
        try std.testing.expectApproxEqRel(c.objective, obj, 1e-9);
    }
}
