"""
KowitoDB Python SDK — Real gRPC client.

Usage:
    from kowitodb import KowitoDBClient

    db = KowitoDBClient("localhost:50051")
    db.remember("OpenAI raised $6.6B in 2024",
                keywords=["openai", "funding"],
                metadata={"company": "OpenAI"})
    results = db.ask("Which companies raised funding?")
    for r in results.results:
        print(f"[{r.relevance_score:.2f}] {r.content}")

    # SQL queries (returns a list of {column: value} dicts)
    rows = db.sql("SELECT * FROM knowledge WHERE metadata.company = 'OpenAI'")
    for row in rows:
        print(row)

    # Update an existing object
    db.update(obj_id, importance=0.9, change_description="bump importance")

    # Agent conversation memory
    db.record_turn("session-1", "user", "What is KowitoDB?")
    turns = db.get_session("session-1")
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
    graph_nodes: int = 0
    graph_edges: int = 0
    active_agent_sessions: int = 0
    total_cost_usd: float = 0.0
    cache_entries: int = 0
    cache_hit_rate: float = 0.0

    @classmethod
    def from_proto(cls, p: pb.StatsResponse) -> "Stats":
        return cls(
            total_objects=p.total_objects,
            vector_count=p.vector_count,
            index_size_bytes=p.index_size_bytes,
            graph_nodes=p.graph_nodes,
            graph_edges=p.graph_edges,
            active_agent_sessions=p.active_agent_sessions,
            total_cost_usd=p.total_cost_usd,
            cache_entries=p.cache_entries,
            cache_hit_rate=p.cache_hit_rate,
        )


@dataclass
class UpdateResult:
    """Result of an update() call."""

    updated: bool
    version: int

    @classmethod
    def from_proto(cls, p: pb.UpdateResponse) -> "UpdateResult":
        return cls(updated=p.updated, version=p.version)


@dataclass
class ConversationTurn:
    """A single turn in an agent conversation session."""

    role: str
    content: str
    timestamp: str = ""

    @classmethod
    def from_proto(cls, p: pb.ConversationTurnProto) -> "ConversationTurn":
        return cls(role=p.role, content=p.content, timestamp=p.timestamp)


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

    def sql(self, query: str) -> List[Dict[str, str]]:
        """Execute a SQL query against the DataFusion engine.

        Returns a list of rows, where each row is a dict mapping column
        name to its string value.

        SELECT * FROM knowledge WHERE metadata.company = 'Acme'
        SELECT content FROM knowledge WHERE keyword LIKE '%enterprise%' LIMIT 10
        """
        self._ensure_connected()
        req = pb.SqlRequest(query=query)
        resp = self._stub.Sql(req)
        return [dict(row.columns) for row in resp.rows]

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

    def update(
        self,
        id: str,
        content: Optional[str] = None,
        metadata: Optional[Dict[str, str]] = None,
        keywords: Optional[List[str]] = None,
        importance: Optional[float] = None,
        change_description: Optional[str] = None,
    ) -> UpdateResult:
        """Update an existing knowledge object.

        Only the provided fields are changed:
        - ``content`` (if set) replaces the content and triggers re-embedding.
        - ``metadata`` is merged into the existing metadata (keys overwrite).
        - ``keywords`` (if non-empty) replaces the keywords.
        - ``importance`` (if set) replaces the importance score.
        - ``change_description`` is recorded in the version history.

        Returns an ``UpdateResult`` with ``updated`` and ``version``.
        """
        self._ensure_connected()
        req = pb.UpdateRequest(id=id, metadata=metadata or {}, keywords=keywords or [])
        if content is not None:
            req.content = content
        if importance is not None:
            req.importance = importance
        if change_description is not None:
            req.change_description = change_description
        resp = self._stub.Update(req)
        return UpdateResult.from_proto(resp)

    def search(self, query: str, top_k: int = 20) -> List[SearchResult]:
        """Direct search (bypasses the AI planner)."""
        self._ensure_connected()
        req = pb.SearchRequest(query=query, top_k=top_k)
        resp = self._stub.Search(req)
        return [SearchResult.from_proto(r) for r in resp.results]

    # ---- Agent conversation memory ----

    def record_turn(self, session_id: str, role: str, content: str) -> int:
        """Record a conversation turn for an agent session.

        ``role`` is one of: user | assistant | system | observation.
        Returns the new total number of turns in the session.
        """
        self._ensure_connected()
        req = pb.RecordTurnRequest(session_id=session_id, role=role, content=content)
        resp = self._stub.RecordTurn(req)
        return resp.turn_count

    def get_session(self, session_id: str) -> Optional[List[ConversationTurn]]:
        """Retrieve all turns for an agent session.

        Returns the list of ``ConversationTurn`` objects, or ``None`` if the
        session does not exist.
        """
        self._ensure_connected()
        req = pb.GetSessionRequest(session_id=session_id)
        resp = self._stub.GetSession(req)
        if not resp.found:
            return None
        return [ConversationTurn.from_proto(t) for t in resp.turns]

    def stats(self) -> Stats:
        """Return database statistics."""
        self._ensure_connected()
        req = pb.StatsRequest()
        resp = self._stub.Stats(req)
        return Stats.from_proto(resp)

    def _ensure_connected(self):
        if self._stub is None:
            self.connect()
