"""LlamaIndex integration for KowitoDB.

Requires ``llama-index-core`` (``pip install "kowitodb[llamaindex]"``).

Exposes :class:`KowitoDBRetriever`, a LlamaIndex ``BaseRetriever`` backed by
KowitoDB's ``ai.ask()`` planner (or raw ``search()``).

Example::

    from kowitodb import KowitoDBClient
    from kowitodb.integrations.llamaindex import KowitoDBRetriever

    client = KowitoDBClient("localhost:50051")
    retriever = KowitoDBRetriever(client, top_k=5)
    nodes = retriever.retrieve("which customers renewed after Series A?")
"""

from __future__ import annotations

from typing import Dict, List, Optional

from llama_index.core.retrievers import BaseRetriever
from llama_index.core.schema import NodeWithScore, QueryBundle, TextNode

from kowitodb import KowitoDBClient


class KowitoDBRetriever(BaseRetriever):
    """LlamaIndex retriever backed by a :class:`KowitoDBClient`.

    Uses the ``ai.ask()`` pipeline by default; set ``use_ask=False`` for the raw
    ``search()`` path. ``metadata_filter`` applies exact-match constraints.
    """

    def __init__(
        self,
        client: KowitoDBClient,
        top_k: int = 10,
        metadata_filter: Optional[Dict[str, str]] = None,
        use_ask: bool = True,
        callback_manager: Optional[object] = None,
    ) -> None:
        self._client = client
        self._top_k = top_k
        self._metadata_filter = metadata_filter
        self._use_ask = use_ask
        super().__init__(callback_manager=callback_manager)

    def _retrieve(self, query_bundle: QueryBundle) -> List[NodeWithScore]:
        query = query_bundle.query_str
        nodes: List[NodeWithScore] = []

        if self._use_ask:
            resp = self._client.ask(
                query,
                max_results=self._top_k,
                metadata_filter=self._metadata_filter,
            )
            for r in resp.results:
                node = TextNode(
                    text=r.content,
                    id_=r.id,
                    metadata={"retrieval_source": r.retrieval_source, **r.metadata},
                )
                nodes.append(NodeWithScore(node=node, score=r.relevance_score))
            return nodes

        for r in self._client.search(
            query, top_k=self._top_k, metadata_filter=self._metadata_filter
        ):
            node = TextNode(text=r.content, id_=r.id, metadata=dict(r.metadata))
            nodes.append(NodeWithScore(node=node, score=r.score))
        return nodes
