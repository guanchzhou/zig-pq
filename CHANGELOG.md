# Changelog

## 0.1.1 - 2026-10-20

- Validate `d > 0`, `nlist > 0`, and `data.len % d == 0` in `ivf.build`, `ivf.fromParts`, `Index.add`, `ivf.search`, and `kmeans.lloydStep`.
- Replace unbounded candidate buffering in `ivf.search` with an in-place bounded max-heap of size `k`, hoist scratch buffers across iterations in `kmeans.train`, and vectorise `kmeans.dist2` with `@Vector(4, f64)`.
- Bound CLI `stdin` reads and add GitHub Actions CI workflow.

## 0.1.0 - 2026-10-07

- First release: k-means++ and Lloyd, product-quantization codebooks with encoding and
  asymmetric distance tables, and IVF-PQ over residuals.
- The `zig-pq` command searches flat PQ or IVF-PQ, with optional exact re-ranking.
- Checked against Faiss 1.15.1 (encoding, flat and IVF search on Faiss's own codebooks) and NumPy
  (a Lloyd step); fuzzed invariants.
