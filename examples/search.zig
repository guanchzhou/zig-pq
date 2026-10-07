const std = @import("std");
const zpq = @import("pq");

pub fn main(init: std.process.Init) !void {
    const gpa = init.gpa;
    var prng = std.Random.DefaultPrng.init(42);
    const r = prng.random();
    const n = 2000;
    const d = 32;
    const data = try gpa.alloc(f32, n * d);
    defer gpa.free(data);
    for (data) |*x| x.* = r.floatNorm(f32);

    // 8 subspaces of 256 centroids: 8 bytes per vector instead of 128.
    var ix = try zpq.ivf.build(gpa, data, d, 16, 8, 8, 10, 0x9e3779b97f4a7c15);
    defer ix.deinit(gpa);

    // A slightly perturbed copy of vector 321 is found among the first ten with 4 of 16 lists probed.
    var query: [d]f32 = undefined;
    for (&query, data[321 * d ..][0..d]) |*q, x| q.* = x + 0.05 * r.floatNorm(f32);
    var out: [10]zpq.ivf.Hit = undefined;
    const k = try zpq.ivf.search(gpa, ix, &query, 4, &out);
    var found = false;
    for (out[0..k]) |h| found = found or h.id == 321;
    std.debug.assert(found);
}
