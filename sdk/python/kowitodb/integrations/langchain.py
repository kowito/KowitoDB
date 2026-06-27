"""LangChain integration for KowitoDB.

Requires ``langchain-core`` (``pip install "kowitodb[langchain]"``).

Exposes:

- :class:`KowitoDBRetriever` — a ``BaseRetriever`` backed by KowitoDB's
  ``ai.ask()`` planner (or raw ``search()``).
- :class:`KowitoDBVectorStore` — a ``VectorStore`` whose ``add_texts`` stores
  objects (server-side embedding) and whose ``similarity_search`` runs search.

Example::

    from kowitodb import KowitoDBClient
    from kowitodb.integrations.langchain import KowitoDBRetriever

    client = KowitoDBClient("localhost:50051")
    retriever = KowitoDBRetriever(client=client, max_results=5)
    docs = retriever.invoke("which customers renewed after Series A?")
"""

from __future__ import annotations

from typing import Any, Dict, Iterable, List, Optional

from langchain_core.callbacks import CallbackManagerForRetrieverRun
from langchain_core.documents import Document
from langchain_core.retrievers import BaseRetriever
from langchain_core.vectorstores import VectorStore
from pydantic import ConfigDict

from kowitodb import KowitoDBClient


def _coerce_metadata(metadata: Optional[Dict[str, Any]]) -> Dict[str, str]:
    """KowitoDB metadata is string→string; coerce values to str."""
    return {str(k): str(v) for k, v in (metadata or {}).items()}


class KowitoDBRetriever(BaseRetriever):
    """LangChain retriever backed by a :class:`KowitoDBClient`.

    By default it uses the ``ai.ask()`` pipeline (intent detection, multi-index
    retrieval, graph traversal, rerank). Set ``use_ask=False`` to use the raw
    ``search()`` path instead.
    """

    client: KowitoDBClient
    max_results: int = 10
    metadata_filter: Optional[Dict[str, str]] = None
    use_ask: bool = True

    model_config = ConfigDict(arbitrary_types_allowed=True)

    def _get_relevant_documents(
        self, query: str, *, run_manager: CallbackManagerForRetrieverRun
    ) -> List[Document]:
        if self.use_ask:
            resp = self.client.ask(
                query,
                max_results=self.max_results,
                metadata_filter=self.metadata_filter,
            )
            return [
                Document(
                    page_content=r.content,
                    metadata={
                        "id": r.id,
                        "score": r.relevance_score,
                        "retrieval_source": r.retrieval_source,
                        **r.metadata,
                    },
                )
                for r in resp.results
            ]

        results = self.client.search(
            query, top_k=self.max_results, metadata_filter=self.metadata_filter
        )
        return [
            Document(
                page_content=r.content,
                metadata={"id": r.id, "score": r.score, **r.metadata},
            )
            for r in results
        ]


class KowitoDBVectorStore(VectorStore):
    """LangChain ``VectorStore`` over KowitoDB.

    Embedding happens server-side, so an ``Embeddings`` object is optional and
    only used if you want LangChain to manage embeddings itself.
    """

    def __init__(self, client: KowitoDBClient, embedding: Any = None) -> None:
        self.client = client
        self._embedding = embedding

    @property
    def embeddings(self) -> Any:
        return self._embedding

    def add_texts(
        self,
        texts: Iterable[str],
        metadatas: Optional[List[dict]] = None,
        **kwargs: Any,
    ) -> List[str]:
        texts = list(texts)
        metadatas = metadatas or [{} for _ in texts]
        items = [
            {"content": text, "metadata": _coerce_metadata(meta)}
            for text, meta in zip(texts, metadatas)
        ]
        return self.client.batch_insert(items)

    def similarity_search(
        self, query: str, k: int = 4, **kwargs: Any
    ) -> List[Document]:
        metadata_filter = kwargs.get("metadata_filter") or kwargs.get("filter")
        results = self.client.search(query, top_k=k, metadata_filter=metadata_filter)
        return [
            Document(
                page_content=r.content,
                metadata={"id": r.id, "score": r.score, **r.metadata},
            )
            for r in results
        ]

    def similarity_search_with_score(
        self, query: str, k: int = 4, **kwargs: Any
    ) -> List[tuple]:
        metadata_filter = kwargs.get("metadata_filter") or kwargs.get("filter")
        results = self.client.search(query, top_k=k, metadata_filter=metadata_filter)
        return [
            (
                Document(
                    page_content=r.content,
                    metadata={"id": r.id, **r.metadata},
                ),
                r.score,
            )
            for r in results
        ]

    @classmethod
    def from_texts(
        cls,
        texts: List[str],
        embedding: Any = None,
        metadatas: Optional[List[dict]] = None,
        *,
        client: Optional[KowitoDBClient] = None,
        address: str = "localhost:50051",
        **kwargs: Any,
    ) -> "KowitoDBVectorStore":
        if client is None:
            client = KowitoDBClient(address)
        store = cls(client, embedding=embedding)
        store.add_texts(texts, metadatas=metadatas)
        return store
