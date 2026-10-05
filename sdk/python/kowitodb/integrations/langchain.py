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

    client = KowitoDBClient("localhost:50051", api_key="...", timeout=30.0)
    retriever = KowitoDBRetriever(client=client, max_results=5)
    # or: KowitoDBRetriever.from_address("localhost:50051", api_key="...")
    docs = retriever.invoke("which customers renewed after Series A?")
"""

from __future__ import annotations

from typing import Any, Dict, Iterable, List, Optional

from langchain_core.callbacks import CallbackManagerForRetrieverRun
from langchain_core.documents import Document
from langchain_core.retrievers import BaseRetriever
from langchain_core.vectorstores import VectorStore
from pydantic import ConfigDict

from kowitodb import DEFAULT_TIMEOUT, KowitoDBClient


def _coerce_metadata(metadata: Optional[Dict[str, Any]]) -> Dict[str, str]:
    """KowitoDB metadata is string→string; coerce values to str."""
    return {str(k): str(v) for k, v in (metadata or {}).items()}


def _coerce_filter(metadata_filter: Optional[Dict[str, Any]]) -> Optional[Dict[str, str]]:
    """Metadata filters are exact string matches; stringify non-string values
    the same way :func:`_coerce_metadata` stored them."""
    if not metadata_filter:
        return None
    return _coerce_metadata(metadata_filter)


def _make_client(
    address: str,
    api_key: Optional[str],
    timeout: Optional[float],
    secure: bool,
) -> KowitoDBClient:
    return KowitoDBClient(address, api_key=api_key, timeout=timeout, secure=secure)


class KowitoDBRetriever(BaseRetriever):
    """LangChain retriever backed by a :class:`KowitoDBClient`.

    By default it uses the ``ai.ask()`` pipeline (intent detection, multi-index
    retrieval, graph traversal, rerank). Set ``use_ask=False`` to use the raw
    ``search()`` path instead.
    """

    client: KowitoDBClient
    max_results: int = 10
    metadata_filter: Optional[Dict[str, Any]] = None
    use_ask: bool = True

    model_config = ConfigDict(arbitrary_types_allowed=True)

    @classmethod
    def from_address(
        cls,
        address: str = "localhost:50051",
        *,
        api_key: Optional[str] = None,
        timeout: Optional[float] = DEFAULT_TIMEOUT,
        secure: bool = False,
        **kwargs: Any,
    ) -> "KowitoDBRetriever":
        """Build a retriever with its own client (``api_key``/``timeout``/``secure``
        are passed to :class:`KowitoDBClient`)."""
        return cls(client=_make_client(address, api_key, timeout, secure), **kwargs)

    def _get_relevant_documents(
        self, query: str, *, run_manager: CallbackManagerForRetrieverRun
    ) -> List[Document]:
        if self.use_ask:
            resp = self.client.ask(
                query,
                max_results=self.max_results,
                metadata_filter=_coerce_filter(self.metadata_filter),
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
            query,
            top_k=self.max_results,
            metadata_filter=_coerce_filter(self.metadata_filter),
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
        *,
        ids: Optional[List[Optional[str]]] = None,
        **kwargs: Any,
    ) -> List[str]:
        """Store ``texts`` (embedded server-side) and return their ids.

        ``ids`` (also what ``add_documents`` passes from ``Document.id``) become
        the stored object ids. KowitoDB ids are UUIDs: a non-UUID id raises
        ``ValueError``; a ``None``/empty id lets the server assign one.
        """
        texts = list(texts)
        metadatas = list(metadatas) if metadatas is not None else [{} for _ in texts]
        if len(metadatas) != len(texts):
            raise ValueError(f"got {len(metadatas)} metadatas for {len(texts)} texts")
        if ids is not None:
            ids = list(ids)
            if len(ids) != len(texts):
                raise ValueError(f"got {len(ids)} ids for {len(texts)} texts")
        else:
            ids = [None] * len(texts)
        items = []
        for text, meta, object_id in zip(texts, metadatas, ids):
            item: Dict[str, Any] = {"content": text, "metadata": _coerce_metadata(meta)}
            if object_id:
                item["id"] = object_id
            items.append(item)
        return self.client.batch_insert(items)

    def similarity_search(
        self, query: str, k: int = 4, **kwargs: Any
    ) -> List[Document]:
        metadata_filter = _coerce_filter(
            kwargs.get("metadata_filter") or kwargs.get("filter")
        )
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
        metadata_filter = _coerce_filter(
            kwargs.get("metadata_filter") or kwargs.get("filter")
        )
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
        api_key: Optional[str] = None,
        timeout: Optional[float] = DEFAULT_TIMEOUT,
        secure: bool = False,
        ids: Optional[List[Optional[str]]] = None,
        **kwargs: Any,
    ) -> "KowitoDBVectorStore":
        """Create a store (and, unless ``client`` is given, a client built from
        ``address``/``api_key``/``timeout``/``secure``) and add ``texts``."""
        if client is None:
            client = _make_client(address, api_key, timeout, secure)
        store = cls(client, embedding=embedding)
        store.add_texts(texts, metadatas=metadatas, ids=ids)
        return store
