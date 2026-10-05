# KowitoDB Python SDK

Python gRPC client for [KowitoDB](../../README.md), the AI Knowledge OS.

## Install

```bash
pip install kowitodb                    # client (pulls in grpcio + protobuf)
pip install "kowitodb[langchain]"       # + LangChain integration
pip install "kowitodb[llamaindex]"      # + LlamaIndex integration
pip install "kowitodb[all]"             # everything
```

## Quick start

```python
from kowitodb import KowitoDBClient

with KowitoDBClient("localhost:50051") as db:
    db.remember("Acme renewed their enterprise license after a Series A.",
                metadata={"company": "Acme"})

    resp = db.ask("which customers renewed after Series A?", max_results=5)
    for r in resp.results:
        print(r.relevance_score, r.content)
```

Core methods: `ask`, `remember`, `insert`, `batch_insert`, `get`, `update`,
`forget`, `search` (both `ask`/`search` accept `metadata_filter`), `list`
(pagination), `sql`, `record_turn`, `get_session`, `stats`.
`insert`/`remember`/`batch_insert` items accept an optional `id` (a UUID string)
to choose the object id; otherwise the server assigns one.

## Authentication, deadlines, TLS

```python
db = KowitoDBClient(
    "db.example.com:50051",
    api_key="...",        # sent as `authorization: Bearer <key>` on every RPC
    timeout=30.0,         # default per-RPC deadline in seconds (None = no deadline)
    secure=True,          # TLS with system roots; or root_certificates=b"...PEM..."
                          # or credentials=grpc.ssl_channel_credentials(...)
)
```

`AsyncKowitoDBClient` takes the same keyword arguments. The API key must match
the server's `--api-key` / `KOWITODB_API_KEY`; without TLS it travels in
plaintext, so use `secure=True` (server `--tls-cert`/`--tls-key`) or a
TLS-terminating proxy outside a trusted network. Channels connect lazily — the
first RPC (e.g. `db.stats()`) is what surfaces an unreachable server
(`grpc.RpcError` with `StatusCode.UNAVAILABLE`).

## LangChain

```python
from kowitodb import KowitoDBClient
from kowitodb.integrations.langchain import KowitoDBRetriever, KowitoDBVectorStore

client = KowitoDBClient("localhost:50051", api_key="...")

# As a retriever (uses the ai.ask() planner by default; use_ask=False for raw search)
retriever = KowitoDBRetriever(client=client, max_results=5)
docs = retriever.invoke("which customers renewed after Series A?")

# As a vector store (embedding happens server-side)
store = KowitoDBVectorStore(client)
store.add_texts(["Acme renewed.", "Globex churned."],
                metadatas=[{"company": "Acme"}, {"company": "Globex"}])
hits = store.similarity_search("renewals", k=3, filter={"company": "Acme"})

# add_documents() keeps Document.id (must be a UUID) as the KowitoDB object id.
# Or build everything from an address:
#   KowitoDBRetriever.from_address("localhost:50051", api_key="...", max_results=5)
#   KowitoDBVectorStore.from_texts(texts, address="localhost:50051", api_key="...")
```

## LlamaIndex

```python
from kowitodb import KowitoDBClient
from kowitodb.integrations.llamaindex import KowitoDBRetriever

client = KowitoDBClient("localhost:50051")
retriever = KowitoDBRetriever(client, top_k=5)
# or: KowitoDBRetriever(address="localhost:50051", api_key="...", top_k=5)
nodes = retriever.retrieve("which customers renewed after Series A?")
```

## Regenerating the gRPC stubs

One command (regenerates from `kowitodb-server/proto/kowitodb.proto` and fixes
the relative import automatically):

```bash
make gen-python            # from the repo root
# or:  bash sdk/python/scripts/gen.sh
```

Requires `pip install "kowitodb[codegen]"` (pins `grpcio-tools==1.81.1`, which
fixes the runtime minimums `grpcio>=1.81.1` / `protobuf>=6.33.5` declared in
`pyproject.toml` — bump them together). See [`scripts/gen.sh`](scripts/gen.sh).
