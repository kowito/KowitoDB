"""Benchmark Milvus on the shared dataset, at HNSW params matched to the others.

Requires `pymilvus` and a running Milvus (see docker-compose). Then:

    python bench_milvus.py dataset.bin [--host localhost --port 19530] \
        --m 16 --efc 128 --efs 32,64,128,256
"""

import argparse
import sys

from bench_common import load_dataset, time_queries, summarize


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("dataset")
    ap.add_argument("--host", default="localhost")
    ap.add_argument("--port", default="19530")
    ap.add_argument("--m", type=int, default=16)
    ap.add_argument("--efc", type=int, default=128)
    ap.add_argument("--efs", default="32,64,128,256")
    args = ap.parse_args()
    ef_list = [int(x) for x in args.efs.split(",")]

    from pymilvus import (
        connections, FieldSchema, CollectionSchema, DataType, Collection, utility,
    )

    ds = load_dataset(args.dataset)
    dim, k, n = ds["dim"], ds["k"], ds["n"]
    name = "bench"

    connections.connect(host=args.host, port=args.port)
    if utility.has_collection(name):
        utility.drop_collection(name)
    schema = CollectionSchema([
        FieldSchema("id", DataType.INT64, is_primary=True, auto_id=False),
        FieldSchema("vec", DataType.FLOAT_VECTOR, dim=dim),
    ])
    coll = Collection(name, schema)

    print(f"# inserting {n} vectors into Milvus ...", file=sys.stderr)
    B = 2000
    vecs = ds["vectors"]
    for start in range(0, n, B):
        end = min(start + B, n)
        coll.insert([list(range(start, end)), [vecs[i].tolist() for i in range(start, end)]])
    coll.flush()
    coll.create_index("vec", {
        "index_type": "HNSW", "metric_type": "COSINE",
        "params": {"M": args.m, "efConstruction": args.efc},
    })
    coll.load()

    print(f"system,ef_search,recall@{k},qps_1thread,p50_us,p95_us")
    for ef in ef_list:
        def search(qv, ef=ef):
            res = coll.search([qv.tolist()], "vec", {"metric_type": "COSINE",
                              "params": {"ef": ef}}, limit=k)
            return [hit.id for hit in res[0]]

        results, lat = time_queries(search, ds["queries"])
        summarize("milvus", ef, results, lat, ds["ground_truth"], k)


if __name__ == "__main__":
    main()
