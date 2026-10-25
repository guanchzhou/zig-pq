# Changelog

## 0.1.2 - 2026-10-25

- Fix a double free in `ivf.build` when `Index.add` fails after `ivf.fromParts`, and a codebook leak when `ivf.fromParts` fails. `ivf.fromParts` takes ownership only on success.
- `Index.add` reserves capacity in both per-list arrays before appending, so ids and codes stay in step on `OutOfMemory`, and returns `error.TooManyVectors` when an id would overflow `u32`.
- `Codebook.encode`, `table`, `adc` and `decode` now return errors instead of asserting: `error.DimensionMismatch` for wrong lengths, `error.InvalidCode` for code bytes `>= ksub`, and `error.InvalidShape` or `error.BitsOutOfRange` for an inconsistent codebook. Callers need `try`.
- `pq.fromCentroids` rejects `d == 0`; `kmeans.nearest` returns centroid 0 for an empty vector instead of dividing by zero.
- `kmeans.train` and `kmeans.lloydStep` return `error.NonFiniteValue` for NaN or infinite input; sizes `k * d` and `nlist * d` use checked multiplication.
- Allocation-failure tests over `kmeans.train`, `pq.train`, `pq.fromCentroids`, `ivf.build`, `ivf.fromParts`, `Index.add` and `ivf.search`.

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
