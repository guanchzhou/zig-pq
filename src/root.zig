//! zig-pq: product quantization for approximate nearest-neighbour search.
//!
//! - `kmeans`: k-means++ seeding and Lloyd iterations.
//! - `pq`: product-quantization codebooks, encoding and asymmetric distance tables.
//! - `ivf`: an inverted-file index over product-quantized residuals (IVF-PQ).

const std = @import("std");

pub const kmeans = @import("kmeans.zig");
pub const pq = @import("pq.zig");
pub const ivf = @import("ivf.zig");

test {
    std.testing.refAllDecls(@This());
    std.testing.refAllDecls(kmeans);
    std.testing.refAllDecls(pq);
    std.testing.refAllDecls(ivf);
    _ = @import("fuzz.zig");
}
