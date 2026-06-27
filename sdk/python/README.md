# KowitoDB Python SDK

Python gRPC client for [KowitoDB](../../README.md), the AI Knowledge OS.

## Install

```bash
pip install "kowitodb[grpc]"            # client + gRPC runtime
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

## LangChain

```python
from kowitodb import KowitoDBClient
from kowitodb.integrations.langchain import KowitoDBRetriever, KowitoDBVectorStore

client = KowitoDBClient("localhost:50051")

# As a retriever (uses the ai.ask() planner by default; use_ask=False for raw search)
retriever = KowitoDBRetriever(client=client, max_results=5)
docs = retriever.invoke("which customers renewed after Series A?")

# As a vector store (embedding happens server-side)
store = KowitoDBVectorStore(client)
store.add_texts(["Acme renewed.", "Globex churned."],
                metadatas=[{"company": "Acme"}, {"company": "Globex"}])
hits = store.similarity_search("renewals", k=3)
```

## LlamaIndex

```python
from kowitodb import KowitoDBClient
from kowitodb.integrations.llamaindex import KowitoDBRetriever

client = KowitoDBClient("localhost:50051")
retriever = KowitoDBRetriever(client, top_k=5)
nodes = retriever.retrieve("which customers renewed after Series A?")
```

## Regenerating the gRPC stubs

```bash
python -m grpc_tools.protoc -I ../../proto \
  --python_out=kowitodb --grpc_python_out=kowitodb \
  ../../proto/kowitodb.proto
```

Then patch the grpc stub's `import kowitodb_pb2` to `from . import kowitodb_pb2`
so the package import works.
