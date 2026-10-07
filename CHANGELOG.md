# Changelog

## 0.1.0 - 2026-10-07

- First release: k-means++ and Lloyd, product-quantization codebooks with encoding and
  asymmetric distance tables, and IVF-PQ over residuals.
- The `zig-pq` command searches flat PQ or IVF-PQ, with optional exact re-ranking.
- Checked against Faiss 1.15.1 (encoding, flat and IVF search on Faiss's own codebooks) and NumPy
  (a Lloyd step); fuzzed invariants.
