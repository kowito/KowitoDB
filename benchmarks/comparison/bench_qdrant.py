"""Benchmark Qdrant on the shared dataset, at HNSW params matched to the others.

Talks to Qdrant over its REST API (only `requests` needed — robust across
Python versions). Bring Qdrant up with the bundled docker-compose, then:

    python bench_qdrant.py dataset.bin [--url http://localhost:6333] \
        --m 16 --efc 128 --efs 32,64,128,256
"""

import argparse
import sys
import time
import requests

from bench_common import load_dataset, time_queries, summarize


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("dataset")
    ap.add_argument("--url", default="http://localhost:6333")
    ap.add_argument("--m", type=int, default=16)
    ap.add_argument("--efc", type=int, default=128)
    ap.add_argument("--efs", default="32,64,128,256")
    args = ap.parse_args()
    ef_list = [int(x) for x in args.efs.split(",")]

    ds = load_dataset(args.dataset)
    dim, k = ds["dim"], ds["k"]
    coll = "bench"
    s = requests.Session()

    # (Re)create the collection with HNSW params matched to KowitoDB/Milvus.
    # indexing_threshold=1 forces Qdrant to build the HNSW index (otherwise small
    # collections stay as a plain segment and do *exact* search → recall 1.0 and
    # ef has no effect, which isn't an HNSW comparison).
    s.delete(f"{args.url}/collections/{coll}")
    r = s.put(f"{args.url}/collections/{coll}", json={
        "vectors": {"size": dim, "distance": "Cosine",
                    "hnsw_config": {"m": args.m, "ef_construct": args.efc}},
        "optimizers_config": {"indexing_threshold": 1},
    })
    r.raise_for_status()

    # Upload all base vectors (batched).
    print(f"# uploading {ds['n']} vectors to Qdrant ...", file=sys.stderr)
    B = 1000
    vecs = ds["vectors"]
    for start in range(0, ds["n"], B):
        end = min(start + B, ds["n"])
        points = [{"id": i, "vector": vecs[i].tolist()} for i in range(start, end)]
        r = s.put(f"{args.url}/collections/{coll}/points?wait=true",
                  json={"points": points})
        r.raise_for_status()

    # Wait for HNSW indexing to finish (status green + all vectors indexed).
    print("# waiting for Qdrant HNSW indexing ...", file=sys.stderr)
    for _ in range(600):
        info = s.get(f"{args.url}/collections/{coll}").json()["result"]
        if info.get("status") == "green" and info.get("indexed_vectors_count", 0) >= ds["n"]:
            break
        time.sleep(1)
    else:
        print("# WARNING: indexing did not complete; results may be exact search",
              file=sys.stderr)

    print(f"system,ef_search,recall@{k},qps_1thread,p50_us,p95_us")
    for ef in ef_list:
        def search(qv, ef=ef):
            r = s.post(f"{args.url}/collections/{coll}/points/search", json={
                "vector": qv.tolist(), "limit": k,
                "params": {"hnsw_ef": ef},
            })
            r.raise_for_status()
            return [p["id"] for p in r.json()["result"]]

        results, lat = time_queries(search, ds["queries"])
        summarize("qdrant", ef, results, lat, ds["ground_truth"], k)


if __name__ == "__main__":
    main()
