"""Shared dataset loader + metrics for the KowitoDB/Qdrant/Milvus comparison.

The dataset is produced by the Rust side
(`cargo run --release -p kowitodb-index --example bench_compare -- dataset.bin`)
so all three systems benchmark the *identical* vectors, queries, and
brute-force ground truth. Binary format (little-endian):

    magic  : b"KVDB"
    n      : u32         number of base vectors
    dim    : u32         dimensionality
    q      : u32         number of query vectors
    k      : u32         neighbors per query in the ground truth
    vectors: n*dim f32
    queries: q*dim f32
    gt     : q*k   u32   ground-truth neighbor indices (into 0..n)
"""

import struct
import time
import numpy as np


def load_dataset(path):
    with open(path, "rb") as f:
        data = f.read()
    assert data[:4] == b"KVDB", "bad dataset magic"
    n, dim, q, k = struct.unpack_from("<IIII", data, 4)
    off = 4 + 16
    vec_n = n * dim
    vectors = np.frombuffer(data, dtype="<f4", count=vec_n, offset=off).reshape(n, dim)
    off += vec_n * 4
    qn = q * dim
    queries = np.frombuffer(data, dtype="<f4", count=qn, offset=off).reshape(q, dim)
    off += qn * 4
    gtn = q * k
    gt = np.frombuffer(data, dtype="<u4", count=gtn, offset=off).reshape(q, k)
    return {
        "n": n, "dim": dim, "q": q, "k": k,
        "vectors": np.ascontiguousarray(vectors),
        "queries": np.ascontiguousarray(queries),
        "ground_truth": gt,
    }


def recall_at_k(results, ground_truth, k):
    """results: list of lists of returned ids (ints). ground_truth: (q, k) array."""
    hits = 0
    for got, truth in zip(results, ground_truth):
        truth_set = set(int(x) for x in truth[:k])
        hits += sum(1 for g in got[:k] if int(g) in truth_set)
    return hits / (len(results) * k)


def time_queries(search_fn, queries, repeat=1):
    """Run `search_fn(query)->ids` over all queries, return (results, latencies_us)."""
    results, lat = [], []
    for _ in range(repeat):
        results = []
        lat = []
        for qv in queries:
            t = time.perf_counter()
            ids = search_fn(qv)
            lat.append((time.perf_counter() - t) * 1e6)
            results.append(ids)
    lat.sort()
    return results, lat


def summarize(system, ef, results, lat, gt, k):
    recall = recall_at_k(results, gt, k)
    mean_us = sum(lat) / len(lat)
    p50 = lat[len(lat) // 2]
    p95 = lat[int(len(lat) * 0.95)]
    qps = 1e6 / mean_us
    print(f"{system},{ef},{recall:.4f},{qps:.0f},{p50:.0f},{p95:.0f}")
    return recall, qps
