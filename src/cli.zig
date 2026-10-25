const std = @import("std");
const zpq = @import("pq");
const build_options = @import("build_options");
const pq = zpq.pq;
const ivf = zpq.ivf;

const usage =
    \\usage: zig-pq COMMAND < input.json
    \\
    \\  search  {"data":[[..]..],"queries":[[..]..],"m":M,"k":K?,"nbits":B?,"nlist":L?,"nprobe":P?,
    \\           "iters":I?,"rerank":R?,"seed":"HEX"?}
    \\          -> {"results":[[i..]..]}   nearest first by ADC distance. nlist 0 (default) is a flat
    \\             product quantizer; nlist > 0 builds IVF-PQ and searches nprobe lists. With R > 0
    \\             the R best ADC candidates are re-ranked by exact distance before the top K are kept.
    \\  version
    \\
;

fn fail(comptime fmt: []const u8, args: anytype) noreturn {
    std.debug.print("zig-pq: " ++ fmt ++ "\n", args);
    std.process.exit(1);
}

fn flatten(arena: std.mem.Allocator, rows: []const []const f32, what: []const u8) !struct { data: []f32, d: usize } {
    if (rows.len == 0) fail("{s} needs at least one row", .{what});
    const d = rows[0].len;
    if (d == 0) fail("{s} rows must not be empty", .{what});
    const data = try arena.alloc(f32, rows.len * d);
    for (rows, 0..) |r, i| {
        if (r.len != d) fail("{s} row {d} has {d} values, expected {d}", .{ what, i, r.len, d });
        @memcpy(data[i * d ..][0..d], r);
    }
    return .{ .data = data, .d = d };
}

fn hitBefore(_: void, a: ivf.Hit, b: ivf.Hit) bool {
    return a.dist2 < b.dist2 or (a.dist2 == b.dist2 and a.id < b.id);
}

pub fn main(init: std.process.Init) !void {
    const arena = init.arena.allocator();
    const args = try init.minimal.args.toSlice(arena);
    var buf: [64 * 1024]u8 = undefined;
    var stdout = std.Io.File.stdout().writerStreaming(init.io, &buf);
    const out = &stdout.interface;
    defer out.flush() catch {};
    if (args.len != 2) {
        std.debug.print("{s}", .{usage});
        std.process.exit(2);
    }
    const cmd: []const u8 = args[1];
    if (std.mem.eql(u8, cmd, "version") or std.mem.eql(u8, cmd, "--version")) {
        try out.print("zig-pq {s}\n", .{build_options.version});
        return;
    }
    if (std.mem.eql(u8, cmd, "help") or std.mem.eql(u8, cmd, "--help") or std.mem.eql(u8, cmd, "-h")) {
        try out.writeAll(usage);
        return;
    }
    if (!std.mem.eql(u8, cmd, "search")) {
        std.debug.print("zig-pq: unknown command: {s}\n{s}", .{ cmd, usage });
        std.process.exit(2);
    }
    var in_buf: [64 * 1024]u8 = undefined;
    var stdin = std.Io.File.stdin().readerStreaming(init.io, &in_buf);
    const text = stdin.interface.allocRemaining(arena, .limited(256 * 1024 * 1024)) catch |e| switch (e) {
        error.ReadFailed => return stdin.err.?,
        else => |x| return x,
    };
    const in = std.json.parseFromSliceLeaky(struct {
        data: []const []const f32,
        queries: []const []const f32,
        m: usize,
        k: usize = 10,
        nbits: u8 = 8,
        nlist: usize = 0,
        nprobe: usize = 1,
        iters: usize = 20,
        rerank: usize = 0,
        seed: ?[]const u8 = null,
    }, arena, text, .{ .ignore_unknown_fields = true }) catch |e| fail("invalid input: {s}", .{@errorName(e)});
    const seed: u64 = if (in.seed) |s| std.fmt.parseInt(u64, s, 16) catch fail("seed must be hexadecimal: {s}", .{s}) else 0x9e3779b97f4a7c15;
    const m = try flatten(arena, in.data, "data");
    const qs = try flatten(arena, in.queries, "queries");
    if (qs.d != m.d) fail("queries have {d} values, data has {d}", .{ qs.d, m.d });
    if (in.k == 0) fail("k must be at least 1", .{});
    const n = in.data.len;
    const depth = @min(n, @max(in.k, in.rerank));
    const results = try arena.alloc([]u32, in.queries.len);
    const hits = try arena.alloc(ivf.Hit, n);
    if (in.nlist == 0) {
        const cb = pq.train(arena, m.data, m.d, in.m, in.nbits, in.iters, seed) catch |e| fail("{s}", .{@errorName(e)});
        const codes = try arena.alloc(u8, n * in.m);
        for (0..n) |i| try cb.encode(m.data[i * m.d ..][0..m.d], codes[i * in.m ..][0..in.m]);
        const tab = try arena.alloc(f32, in.m * cb.ksub());
        for (results, 0..) |*res, qi| {
            try cb.table(qs.data[qi * m.d ..][0..m.d], tab);
            for (hits, 0..) |*h, i| h.* = .{ .id = @intCast(i), .dist2 = try cb.adc(tab, codes[i * in.m ..][0..in.m]) };
            std.mem.sort(ivf.Hit, hits, {}, hitBefore);
            res.* = try finish(arena, hits[0..depth], in.k, in.rerank, m.data, m.d, qs.data[qi * m.d ..][0..m.d]);
        }
    } else {
        const ix = ivf.build(arena, m.data, m.d, in.nlist, in.m, in.nbits, in.iters, seed) catch |e| fail("{s}", .{@errorName(e)});
        for (results, 0..) |*res, qi| {
            const got = try ivf.search(arena, ix, qs.data[qi * m.d ..][0..m.d], in.nprobe, hits[0..depth]);
            res.* = try finish(arena, hits[0..got], in.k, in.rerank, m.data, m.d, qs.data[qi * m.d ..][0..m.d]);
        }
    }
    try std.json.Stringify.value(.{ .results = results }, .{}, out);
    try out.writeAll("\n");
}

fn finish(arena: std.mem.Allocator, cand: []ivf.Hit, k: usize, rerank: usize, data: []const f32, d: usize, q: []const f32) ![]u32 {
    if (rerank > 0) {
        for (cand) |*h| h.dist2 = @floatCast(zpq.kmeans.dist2(data[@as(usize, h.id) * d ..][0..d], q));
        std.mem.sort(ivf.Hit, cand, {}, hitBefore);
    }
    const ids = try arena.alloc(u32, @min(k, cand.len));
    for (ids, cand[0..ids.len]) |*o, h| o.* = h.id;
    return ids;
}
