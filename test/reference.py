"""Reference outputs for zig-pq, written to test/golden.json.

Faiss (faiss-cpu) is the reference for encoding, asymmetric-distance search and IVF-PQ search:
it trains its own codebooks and coarse quantizers, and the Zig side loads exactly those, so the
comparison checks encoding and search against Faiss rather than two trainings against each other.
NumPy is the reference for one Lloyd step of k-means from given initial centroids.

Run: uv run --python 3.12 --with numpy --with faiss-cpu python test/reference.py
"""

import json
from pathlib import Path

import faiss
import numpy as np

OUT = Path(__file__).resolve().parent / "golden.json"
RNG = np.random.default_rng(20261007)


def flat(a) -> list:
    return np.asarray(a, dtype=np.float64).ravel().tolist()


def data(n, d):
    # Clustered data, so that codebooks and lists have structure.
    centers = RNG.normal(size=(8, d)) * 3
    x = centers[RNG.integers(0, 8, size=n)] + RNG.normal(size=(n, d))
    return x.astype(np.float32)


def pq_case():
    d = int(RNG.choice([16, 32, 64]))
    m = int(RNG.choice([4, 8]))
    n = int(RNG.integers(300, 700))
    x = data(n, d)
    q = data(5, d)
    index = faiss.IndexPQ(d, m, 8)
    index.train(x)
    index.add(x)
    cents = faiss.vector_to_array(index.pq.centroids)
    codes = index.pq.compute_codes(x)
    D, I = index.search(q, 10)
    return {"d": d, "m": m, "nbits": 8, "data": flat(x), "queries": flat(q), "centroids": flat(cents),
            "codes": codes.ravel().tolist(), "dist": flat(D), "ids": I.ravel().tolist()}


def ivf_case():
    d = int(RNG.choice([16, 32]))
    m = int(RNG.choice([4, 8]))
    nlist = int(RNG.choice([4, 8, 16]))
    nprobe = int(RNG.integers(1, nlist + 1))
    n = int(RNG.integers(600, 1200))
    x = data(n, d)
    q = data(5, d)
    quantizer = faiss.IndexFlatL2(d)
    index = faiss.IndexIVFPQ(quantizer, d, nlist, m, 8)
    index.train(x)
    index.add(x)
    index.nprobe = nprobe
    D, I = index.search(q, 10)
    return {"d": d, "m": m, "nbits": 8, "nlist": nlist, "nprobe": nprobe, "data": flat(x), "queries": flat(q),
            "coarse": flat(quantizer.reconstruct_n(0, nlist)), "centroids": flat(faiss.vector_to_array(index.pq.centroids)),
            "dist": flat(D), "ids": I.ravel().tolist()}


def lloyd_case():
    d = int(RNG.integers(2, 12))
    k = int(RNG.integers(2, 20))
    n = int(RNG.integers(k + 5, 300))
    x = data(n, d)
    init = x[RNG.choice(n, size=k, replace=False)].copy()
    X = x.astype(np.float64)
    C = init.astype(np.float64)
    dist = ((X[:, None, :] - C[None, :, :]) ** 2).sum(axis=2)
    assign = dist.argmin(axis=1)
    new = C.copy()
    for c in range(k):
        pts = X[assign == c]
        if len(pts):
            new[c] = pts.mean(axis=0)
    return {"d": d, "k": k, "data": flat(x), "init": flat(init), "assign": assign.tolist(),
            "centroids": flat(new.astype(np.float32)), "objective": float(dist.min(axis=1).sum())}


def main():
    golden = {"faiss": faiss.__version__, "pq": [pq_case() for _ in range(8)], "ivf": [ivf_case() for _ in range(6)],
              "lloyd": [lloyd_case() for _ in range(12)]}
    OUT.write_text(json.dumps(golden, separators=(",", ":")) + "\n")
    print(f"wrote {OUT}: faiss {golden['faiss']}, {sum(len(v) for k, v in golden.items() if k != 'faiss')} cases")


if __name__ == "__main__":
    main()
