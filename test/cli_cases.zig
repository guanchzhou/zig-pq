//! Command-line cases run by `zig build test-cli`. `stdout` is compared
//! exactly, `stderr` as a substring.

pub const Case = struct {
    name: []const u8,
    args: []const []const u8,
    stdin: ?[]const u8 = null,
    stdout: ?[]const u8 = null,
    stderr: ?[]const u8 = null,
    exit: u8 = 0,
};

pub const cases = [_]Case{
    .{
        .name = "search: flat product quantizer",
        .args = &.{"search"},
        .stdin = "{\"data\":[[0,0,0,0],[0.1,0,0,0],[5,5,5,5],[5.1,5,5,5],[0,9,0,9],[0.1,9,0,9]],\"queries\":[[5,5,5,5.05]],\"m\":2,\"nbits\":1,\"k\":2,\"iters\":5}",
        .stdout = "{\"results\":[[2,3]]}\n",
    },
    .{
        .name = "search: IVF-PQ with exact re-rank",
        .args = &.{"search"},
        .stdin = "{\"data\":[[0,0,0,0],[0.1,0,0,0],[5,5,5,5],[5.1,5,5,5],[0,9,0,9],[0.1,9,0,9]],\"queries\":[[5,5,5,5.05]],\"m\":2,\"nbits\":1,\"k\":2,\"nlist\":2,\"nprobe\":2,\"iters\":5,\"rerank\":6}",
        .stdout = "{\"results\":[[2,3]]}\n",
    },
    .{ .name = "error: width not divisible by m", .args = &.{"search"}, .stdin = "{\"data\":[[1,2,3]],\"queries\":[[1,2,3]],\"m\":2}", .stderr = "InvalidShape", .exit = 1 },
    .{ .name = "error: query width", .args = &.{"search"}, .stdin = "{\"data\":[[1,2]],\"queries\":[[1]],\"m\":1}", .stderr = "queries have 1 values, data has 2", .exit = 1 },
    .{ .name = "error: truncated JSON", .args = &.{"search"}, .stdin = "{", .stderr = "invalid input", .exit = 1 },
    .{ .name = "error: unknown command", .args = &.{"bogus"}, .stderr = "unknown command: bogus", .exit = 2 },
    .{ .name = "error: no command prints usage", .args = &.{}, .stderr = "usage: zig-pq COMMAND", .exit = 2 },
    .{ .name = "help: exits 0", .args = &.{"--help"} },
};
