# zig-pq

Product quantization for approximate nearest-neighbour search, in pure Zig 0.17: a vector is
stored as one byte per subspace, distances are estimated from small lookup tables, and an
inverted-file index narrows the search to a few lists.

- **k-means**: k-means++ seeding from a seeded stream, Lloyd iterations, f64 accumulation, ties
  to the lower index; a centroid that loses its points keeps its place.
- **Product quantization** (Jégou, Douze and Schmid 2011): `m` subspace codebooks of `2^nbits`
  centroids; encoding, decoding, and asymmetric distances (ADC) through a per-query table.
- **IVF-PQ**: a coarse quantizer with product-quantized residuals; a query visits the `nprobe`
  nearest lists.
- No dependencies and no C. Checked against Faiss for encoding and search, and against NumPy for
  k-means. MIT licensed.

## Install

```sh
zig fetch --save git+https://github.com/guanchzhou/zig-pq
```

```zig
const zpq = b.dependency("zig_pq", .{}).module("pq");
exe.root_module.addImport("pq", zpq);
```

## Library

```zig
const cb = try zpq.pq.train(gpa, data, d, m, 8, 20, seed); // m codebooks of 256 centroids
defer cb.deinit(gpa);
cb.encode(x, code);              // m bytes
cb.table(query, tab);            // m * 256 squared distances
const d2 = cb.adc(tab, code);    // estimated squared distance

var ix = try zpq.ivf.build(gpa, data, d, nlist, m, 8, 20, seed);
defer ix.deinit(gpa);
_ = try zpq.ivf.search(gpa, ix, query, nprobe, hits); // nearest first by ADC distance
```

`pq.fromCentroids` and `ivf.fromParts` load codebooks and coarse centroids trained elsewhere, for
example by Faiss. [`examples/search.zig`](examples/search.zig) builds IVF-PQ over 2,000 random
vectors at 8 bytes each and finds a perturbed copy of one of them with 4 of 16 lists probed.

## Command line

`zig build` installs `zig-out/bin/zig-pq`. `search` reads one JSON object on stdin, trains on the
data, and writes the nearest ids for each query:

```sh
$ echo '{"data":[[0,0,0,0],[0.1,0,0,0],[5,5,5,5],[5.1,5,5,5],[0,9,0,9],[0.1,9,0,9]],"queries":[[5,5,5,5.05]],"m":2,"nbits":1,"k":2,"iters":5}' | zig-pq search
{"results":[[2,3]]}
```

`nlist` > 0 builds IVF-PQ; `rerank` re-ranks the best ADC candidates by exact distance.
`zig-pq --help` lists the fields.

## Development

```sh
zig build test              # unit, golden, fuzz-corpus, example, and command-line tests
zig build test --fuzz       # fuzz the invariants (needs .zig-cache/tmp to exist)
zig build docs              # API documentation in zig-out/docs
uv run --python 3.12 --with numpy --with faiss-cpu python test/reference.py   # regenerate test/golden.json
```

`test/reference.py` trains Faiss's own `IndexPQ` and `IndexIVFPQ` and records their codebooks,
codes and search results; `test/golden.zig` loads those codebooks and must reproduce Faiss's
codes (differences allowed only between centroids within 1e-5 of each other in distance, since
Faiss computes in float32) and its ranked distances within 1e-4. A relative change of 1e-3 in
the ADC table makes it fail. One Lloyd step is checked against NumPy from the same initial
centroids: identical assignments, centroids within 1e-5.

## Credits

- H. Jégou, M. Douze and C. Schmid, "Product Quantization for Nearest Neighbor Search", *IEEE
  TPAMI* 33(1), 117-128 (2011).
- D. Arthur and S. Vassilvitskii, "k-means++: The Advantages of Careful Seeding", *SODA* (2007).
- M. Douze et al., "The Faiss library" (2024), used here as the reference implementation only.

## License

MIT. See [LICENSE](LICENSE).
