#!/usr/bin/env python3
"""
KowitoDB Demo — Enterprise Knowledge Retrieval

Connects to a running KowitoDB gRPC server, inserts sample knowledge,
and runs natural-language queries.

Usage:
    # Terminal 1: Start the server
    cargo run -p kowitodb -- serve

    # Terminal 2: Run the demo
    source .venv/bin/activate
    python examples/demo_enterprise.py
"""

import os
import sys
import time

sys.path.insert(0, os.path.join(os.path.dirname(__file__), "..", "sdk", "python"))

import grpc  # noqa: E402

from kowitodb import KowitoDBClient  # noqa: E402

ADDRESS = os.environ.get("KOWITODB_ADDRESS", "localhost:50051")


def main():
    print("=" * 60)
    print("  KowitoDB — Enterprise Knowledge Retrieval Demo")
    print("=" * 60)
    print()

    # Connect. gRPC channels connect lazily, so make a real (cheap) RPC to find
    # out whether the server is reachable.
    print(f"Connecting to KowitoDB server at {ADDRESS}...")
    db = KowitoDBClient(
        ADDRESS, api_key=os.environ.get("KOWITODB_API_KEY"), timeout=10.0
    )

    try:
        stats = db.stats()
    except grpc.RpcError as e:
        print(f"⚠️  Could not connect to server: {e.code().name}: {e.details()}")
        print()
        if e.code() == grpc.StatusCode.UNAUTHENTICATED:
            print("The server requires an API key: set KOWITODB_API_KEY.")
        else:
            print("Start the server first:")
            print("  cargo run -p kowitodb -- serve")
        print()
        print("Then run this demo again.")
        db.close()
        return

    print("✅ Connected.")
    print(
        f"📊 Current state: {stats.total_objects} objects, {stats.vector_count} vectors"
    )
    print()

    # Insert sample knowledge
    print("Step 1: Inserting sample enterprise knowledge...")
    print()

    companies = [
        {
            "content": "Acme Corp renewed their enterprise license in March 2024 after raising Series A funding of $15M from Sequoia.",
            "keywords": [
                "acme",
                "renewal",
                "series-a",
                "enterprise",
                "funding",
                "sequoia",
            ],
            "metadata": {
                "company": "Acme Corp",
                "stage": "series_a",
                "renewed": "true",
                "funding_amount": "15M",
            },
        },
        {
            "content": "Globex Inc. received Series B funding of $30M in January 2024 led by a16z and upgraded to enterprise tier.",
            "keywords": [
                "globex",
                "series-b",
                "enterprise",
                "upgrade",
                "funding",
                "a16z",
            ],
            "metadata": {
                "company": "Globex Inc.",
                "stage": "series_b",
                "renewed": "true",
                "funding_amount": "30M",
            },
        },
        {
            "content": "Initech went through Series A in Q3 2023 but churned in December 2024 due to budget cuts.",
            "keywords": ["initech", "series-a", "churn", "budget"],
            "metadata": {
                "company": "Initech",
                "stage": "series_a",
                "renewed": "false",
                "funding_amount": "8M",
            },
        },
        {
            "content": "Umbrella Corp raised $50M Series A in Q3 2024 from Lightspeed and signed a 3-year enterprise agreement.",
            "keywords": [
                "umbrella",
                "series-a",
                "enterprise",
                "agreement",
                "funding",
                "lightspeed",
            ],
            "metadata": {
                "company": "Umbrella Corp",
                "stage": "series_a",
                "renewed": "true",
                "funding_amount": "50M",
            },
        },
        {
            "content": "Cyberdyne Systems completed Series C at $200M valuation and maintains enterprise subscription since 2022.",
            "keywords": ["cyberdyne", "series-c", "enterprise", "subscription"],
            "metadata": {
                "company": "Cyberdyne Systems",
                "stage": "series_c",
                "renewed": "true",
                "funding_amount": "200M",
            },
        },
    ]

    for c in companies:
        try:
            obj_id = db.remember(
                content=c["content"],
                keywords=c["keywords"],
                metadata=c["metadata"],
                importance=0.8,
            )
            print(f"  ✅ {c['metadata']['company']} → {obj_id[:8]}...")
        except Exception as e:
            print(f"  ❌ Failed to insert {c['metadata']['company']}: {e}")

    print()
    print(f"  Inserted {len(companies)} company records.")
    print()

    # Step 2: Natural language queries
    print("Step 2: Running ai.ask() queries...")
    print()

    queries = [
        "Which enterprise customers renewed after Series A?",
        "Compare Series A and Series B companies",
        "Who raised the most funding?",
        "List all companies that churned",
        "Which companies renewed after January 2024?",
    ]

    for q in queries:
        print(f"  👤 {q}")
        try:
            start = time.time()
            resp = db.ask(q, max_results=3)
            elapsed = (time.time() - start) * 1000

            print(f"  🤖 Intent: {resp.detected_intent} ({elapsed:.1f}ms)")
            for i, r in enumerate(resp.results):
                preview = r.content[:100]
                print(
                    f"     {i + 1}. [{r.relevance_score:.2f}] [{r.retrieval_source}] {preview}..."
                )
            if not resp.results:
                print("     (no results)")
        except Exception as e:
            print(f"  ❌ Error: {e}")
        print()

    # Step 3: SQL query (DataFusion). `metadata` is a JSON-encoded string column,
    # so match keys/values with LIKE. Rows come back as {column: value} dicts.
    print("Step 3: SQL query...")
    print()
    query = (
        "SELECT id, content, importance FROM knowledge "
        "WHERE metadata LIKE '%\"stage\":\"series_a\"%' "
        "AND metadata LIKE '%\"renewed\":\"true\"%' "
        "ORDER BY importance DESC"
    )
    print(f"  {query}")
    try:
        rows = db.sql(query)
        for row in rows:
            print(
                f"  {row['id'][:8]}... [importance {float(row['importance']):.2f}] "
                f"{row['content'][:80]}..."
            )
        if not rows:
            print("  (no results)")
    except grpc.RpcError as e:
        print(f"  ❌ Error: {e.details()}")
    print()

    # Step 4: Stats
    print("Step 4: Database stats...")
    print()
    try:
        stats = db.stats()
        print(f"  Total objects:    {stats.total_objects}")
        print(f"  Vector count:     {stats.vector_count}")
        print(f"  Index size bytes: {stats.index_size_bytes}")
    except Exception as e:
        print(f"  ❌ Error: {e}")
    print()

    print("=" * 60)
    print("  Demo complete. 🎉")
    print()
    print("  Try your own queries:")
    print('    cargo run -p kowitodb -- ask "<your question>"')
    print('    cargo run -p kowitodb -- sql "SELECT * FROM knowledge WHERE ..."')
    print("=" * 60)
    db.close()


if __name__ == "__main__":
    main()
