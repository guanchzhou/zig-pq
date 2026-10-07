//! k-means for codebook training: k-means++ seeding from a seeded SplitMix64 stream, then Lloyd
//! iterations. Squared Euclidean distances accumulate in f64; ties go to the lower centroid index;
//! a centroid that loses all its points keeps its previous position.

const std = @import("std");

pub const Error = error{ OutOfMemory, TooFewPoints, DimensionMismatch };

pub fn splitmix64(seed: u64, i: u64) u64 {
    var z = seed +% (i +% 1) *% 0x9e3779b97f4a7c15;
    z = (z ^ (z >> 30)) *% 0xbf58476d1ce4e5b9;
    z = (z ^ (z >> 27)) *% 0x94d049bb133111eb;
    return z ^ (z >> 31);
}

fn uniform(seed: u64, i: u64) f64 {
    return @as(f64, @floatFromInt(splitmix64(seed, i) >> 11)) * 0x1p-53;
}

pub fn dist2(a: []const f32, b: []const f32) f64 {
    var s: f64 = 0;
    for (a, b) |x, y| {
        const t = @as(f64, x) - y;
        s += t * t;
    }
    return s;
}

/// Index of the centroid nearest to `x` among `k` row-major centroids of width `x.len`.
pub fn nearest(centroids: []const f32, x: []const f32) struct { index: u32, dist2: f64 } {
    const d = x.len;
    const k = centroids.len / d;
    var best: u32 = 0;
    var best_d: f64 = std.math.inf(f64);
    for (0..k) |c| {
        const v = dist2(centroids[c * d ..][0..d], x);
        if (v < best_d) {
            best_d = v;
            best = @intCast(c);
        }
    }
    return .{ .index = best, .dist2 = best_d };
}

/// One Lloyd iteration: assigns every point, then moves each centroid to the mean of its points.
/// Returns the objective (sum of squared distances) of the assignment made.
pub fn lloydStep(gpa: std.mem.Allocator, data: []const f32, d: usize, centroids: []f32, assign: []u32) Error!f64 {
    const n = data.len / d;
    const k = centroids.len / d;
    if (assign.len != n) return error.DimensionMismatch;
    var objective: f64 = 0;
    for (0..n) |i| {
        const r = nearest(centroids, data[i * d ..][0..d]);
        assign[i] = r.index;
        objective += r.dist2;
    }
    const sums = try gpa.alloc(f64, k * d);
    defer gpa.free(sums);
    const counts = try gpa.alloc(u32, k);
    defer gpa.free(counts);
    @memset(sums, 0);
    @memset(counts, 0);
    for (0..n) |i| {
        const c = assign[i];
        counts[c] += 1;
        for (sums[c * d ..][0..d], data[i * d ..][0..d]) |*s, x| s.* += x;
    }
    for (0..k) |c| {
        if (counts[c] == 0) continue;
        const inv = 1.0 / @as(f64, @floatFromInt(counts[c]));
        for (centroids[c * d ..][0..d], sums[c * d ..][0..d]) |*v, s| v.* = @floatCast(s * inv);
    }
    return objective;
}

/// Trains `k` centroids of width `d` on `data` with k-means++ seeding and `iters` Lloyd iterations.
pub fn train(gpa: std.mem.Allocator, data: []const f32, d: usize, k: usize, iters: usize, seed: u64, out: []f32) Error!f64 {
    if (d == 0 or data.len % d != 0 or out.len != k * d) return error.DimensionMismatch;
    const n = data.len / d;
    if (n < k or k == 0) return error.TooFewPoints;
    const closest = try gpa.alloc(f64, n);
    defer gpa.free(closest);
    var draw: u64 = 0;
    const first: usize = @min(n - 1, @as(usize, @intFromFloat(uniform(seed, draw) * @as(f64, @floatFromInt(n)))));
    draw += 1;
    @memcpy(out[0..d], data[first * d ..][0..d]);
    for (0..n) |i| closest[i] = dist2(out[0..d], data[i * d ..][0..d]);
    for (1..k) |c| {
        var total: f64 = 0;
        for (closest) |v| total += v;
        var pick: usize = 0;
        if (total > 0) {
            var target = uniform(seed, draw) * total;
            for (closest, 0..) |v, i| {
                pick = i;
                if (target < v) break;
                target -= v;
            }
        } else pick = c % n;
        draw += 1;
        @memcpy(out[c * d ..][0..d], data[pick * d ..][0..d]);
        for (0..n) |i| closest[i] = @min(closest[i], dist2(out[c * d ..][0..d], data[i * d ..][0..d]));
    }
    const assign = try gpa.alloc(u32, n);
    defer gpa.free(assign);
    var objective: f64 = 0;
    for (0..iters) |_| objective = try lloydStep(gpa, data, d, out, assign);
    return objective;
}

test "two clear clusters" {
    const gpa = std.testing.allocator;
    const data = [_]f32{ 0, 0, 0.1, 0, 0, 0.1, 10, 10, 10.1, 10, 10, 10.1 };
    var c: [4]f32 = undefined;
    _ = try train(gpa, &data, 2, 2, 5, 7, &c);
    const a = nearest(&c, &.{ 0, 0 }).index;
    const b = nearest(&c, &.{ 10, 10 }).index;
    try std.testing.expect(a != b);
    try std.testing.expectApproxEqAbs(@as(f32, 0.0333333), c[a * 2], 1e-5);
    try std.testing.expectApproxEqAbs(@as(f32, 10.0333333), c[b * 2], 1e-5);
}

test "the objective never increases" {
    const gpa = std.testing.allocator;
    var prng = std.Random.DefaultPrng.init(4);
    const r = prng.random();
    var data: [300 * 4]f32 = undefined;
    for (&data) |*x| x.* = r.floatNorm(f32);
    var c: [16 * 4]f32 = undefined;
    _ = try train(gpa, &data, 4, 16, 0, 9, &c);
    var assign: [300]u32 = undefined;
    var prev = std.math.inf(f64);
    for (0..10) |_| {
        const o = try lloydStep(gpa, &data, 4, &c, &assign);
        try std.testing.expect(o <= prev + 1e-9);
        prev = o;
    }
}
