"""
KowitoDB Python SDK — Real gRPC client.

Usage:
    from kowitodb import KowitoDBClient

    db = KowitoDBClient("localhost:50051")
    db.remember("OpenAI raised $6.6B in 2024",
                keywords=["openai", "funding"],
                metadata={"company": "OpenAI"})
    results = db.ask("Which companies raised funding?")
    for r in results:
        print(f"[{r.relevance_score:.2f}] {r.content}")

    # SQL queries
    rows = db.sql("SELECT * FROM knowledge WHERE metadata.company = 'OpenAI'")
"""

from dataclasses import dataclass, field
from typing import Dict, List, Optional

import grpc

from . import kowitodb_pb2 as pb
from . import kowitodb_pb2_grpc as pb_grpc


@dataclass
class AskResult:
    """A single result from ai.ask()."""

    id: str
    content: str
    relevance_score: float
    retrieval_source: str = ""
    metadata: Dict[str, str] = field(default_factory=dict)

    @classmethod
    def from_proto(cls, p: pb.AskResult) -> "AskResult":
        return cls(
            id=p.id,
            content=p.content,
            relevance_score=p.relevance_score,
            retrieval_source=p.retrieval_source,
            metadata=dict(p.metadata),
        )


@dataclass
class AskResponse:
    """Response from ai.ask()."""

    results: List[AskResult]
    plan_explanation: str
    detected_intent: str

    @classmethod
    def from_proto(cls, p: pb.AskResponse) -> "AskResponse":
        return cls(
            results=[AskResult.from_proto(r) for r in p.results],
            plan_explanation=p.plan_explanation,
            detected_intent=p.detected_intent,
        )


@dataclass
class SearchResult:
    """A single search result."""

    id: str
    content: str
    score: float
    metadata: Dict[str, str] = field(default_factory=dict)

    @classmethod
    def from_proto(cls, p: pb.SearchResult) -> "SearchResult":
        return cls(id=p.id, content=p.content, score=p.score, metadata=dict(p.metadata))


@dataclass
class Stats:
    """Database statistics."""

    total_objects: int = 0
    vector_count: int = 0
    index_size_bytes: int = 0

    @classmethod
    def from_proto(cls, p: pb.StatsResponse) -> "Stats":
        return cls(
            total_objects=p.total_objects,
            vector_count=p.vector_count,
            index_size_bytes=p.index_size_bytes,
        )


class KowitoDBClient:
    """Python gRPC client for KowitoDB.

    Usage:
        db = KowitoDBClient("localhost:50051")
        db.remember("Some knowledge to store")
        response = db.ask("What do you know about X?")
    """

    def __init__(self, address: str = "localhost:50051"):
        self.address = address
        self._channel: Optional[grpc.Channel] = None
        self._stub: Optional[pb_grpc.KowitoDBStub] = None

    # ---- Context manager ----

    def __enter__(self):
        self.connect()
        return self

    def __exit__(self, *args):
        self.close()

    # ---- Connection ----

    def connect(self):
        """Establish the gRPC connection."""
        if self._channel is not None:
            return
        self._channel = grpc.insecure_channel(self.address)
        self._stub = pb_grpc.KowitoDBStub(self._channel)

    def close(self):
        """Close the gRPC connection."""
        if self._channel is not None:
            self._channel.close()
            self._channel = None
            self._stub = None

    # ---- High-level AI API ----

    def ask(self, question: str, max_results: int = 10) -> AskResponse:
        """ai.ask() — natural-language query with automatic retrieval.

        The engine detects intent, chooses retrieval strategies,
        searches all indexes, reranks, and returns optimized results.
        """
        self._ensure_connected()
        req = pb.AskRequest(question=question, max_results=max_results)
        resp = self._stub.Ask(req)
        return AskResponse.from_proto(resp)

    def remember(
        self,
        content: str,
        keywords: Optional[List[str]] = None,
        metadata: Optional[Dict[str, str]] = None,
        importance: float = 0.5,
    ) -> str:
        """ai.remember() — store knowledge for future retrieval.

        Returns the object ID.
        """
        self._ensure_connected()
        req = pb.RememberRequest(
            content=content,
            keywords=keywords or [],
            metadata=metadata or {},
            importance=importance,
        )
        resp = self._stub.Remember(req)
        return resp.id

    def forget(self, object_id: str) -> bool:
        """Remove a knowledge object by ID."""
        self._ensure_connected()
        req = pb.DeleteRequest(id=object_id)
        resp = self._stub.Delete(req)
        return resp.existed

    # ---- SQL API ----

    def sql(self, query: str) -> List[AskResult]:
        """Execute a SQL query against knowledge objects.

        SELECT * FROM knowledge WHERE metadata.company = 'Acme'
        SELECT content FROM knowledge WHERE keyword LIKE '%enterprise%' LIMIT 10
        """
        # Route SQL through the search interface for now
        self._ensure_connected()
        req = pb.SearchRequest(query=query, top_k=20)
        resp = self._stub.Search(req)
        return [
            AskResult(
                id=r.id,
                content=r.content,
                relevance_score=r.score,
                metadata=dict(r.metadata),
            )
            for r in resp.results
        ]

    # ---- Low-level API ----

    def insert(
        self,
        content: str,
        keywords: Optional[List[str]] = None,
        metadata: Optional[Dict[str, str]] = None,
        relationships: Optional[List[tuple]] = None,
        importance: float = 0.5,
    ) -> str:
        """Insert a knowledge object explicitly."""
        self._ensure_connected()
        rels = [
            pb.RelationshipInput(relation_type=r[0], target_id=r[1])
            for r in (relationships or [])
        ]
        req = pb.InsertRequest(
            content=content,
            keywords=keywords or [],
            metadata=metadata or {},
            relationships=rels,
            importance=importance,
        )
        resp = self._stub.Insert(req)
        return resp.id

    def get(self, object_id: str) -> Optional[dict]:
        """Retrieve a knowledge object by ID."""
        self._ensure_connected()
        req = pb.GetRequest(id=object_id)
        resp = self._stub.Get(req)
        if resp.HasField("object"):
            o = resp.object
            return {
                "id": o.id,
                "content": o.content,
                "keywords": list(o.keywords),
                "metadata": dict(o.metadata),
                "importance": o.importance,
                "created_at": o.created_at,
            }
        return None

    def search(self, query: str, top_k: int = 20) -> List[SearchResult]:
        """Direct search (bypasses the AI planner)."""
        self._ensure_connected()
        req = pb.SearchRequest(query=query, top_k=top_k)
        resp = self._stub.Search(req)
        return [SearchResult.from_proto(r) for r in resp.results]

    def stats(self) -> Stats:
        """Return database statistics."""
        self._ensure_connected()
        req = pb.StatsRequest()
        resp = self._stub.Stats(req)
        return Stats.from_proto(resp)

    def _ensure_connected(self):
        if self._stub is None:
            self.connect()
