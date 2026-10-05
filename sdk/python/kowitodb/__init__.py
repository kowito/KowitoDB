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

    # Insert many objects at once -> list of ids
    ids = db.batch_insert([
        {"content": "Anthropic raised $4B from Amazon",
         "keywords": ["anthropic", "funding"],
         "metadata": {"company": "Anthropic"}, "importance": 0.8},
        {"content": "Mistral raised €600M",
         "metadata": {"company": "Mistral"}},
    ])

    # Scroll through stored objects (paginated)
    page = db.list(offset=0, limit=50)
    print(f"{len(page.objects)} of {page.total} objects")

    # Metadata-filtered ask / search (exact-match, ANDed)
    results = db.ask("funding rounds", metadata_filter={"company": "Anthropic"})
    hits = db.search("funding", top_k=10, metadata_filter={"company": "Anthropic"})

    # SQL queries (DataFusion; returns a list of {column: value} dicts).
    # `metadata`/`keywords` are JSON-encoded string columns — match with LIKE.
    rows = db.sql('''
        SELECT id, content FROM knowledge
        WHERE metadata LIKE '%"company":"OpenAI"%'
    ''')
    for row in rows:
        print(row)

    # Update an existing object
    db.update(obj_id, importance=0.9, change_description="bump importance")

    # Agent conversation memory
    db.record_turn("session-1", "user", "What is KowitoDB?")
    turns = db.get_session("session-1")

Authentication, deadlines and TLS:

    db = KowitoDBClient(
        "db.example.com:50051",
        api_key="...",      # sent as `authorization: Bearer <key>` on every RPC
        timeout=30.0,       # default per-RPC deadline in seconds (None = no deadline)
        secure=True,        # TLS using the system roots (or pass root_certificates=
                            # / credentials=grpc.ssl_channel_credentials(...))
    )
"""

import collections
import uuid
from dataclasses import dataclass, field
from typing import Any, Dict, List, Optional, Sequence, Tuple

import grpc
import grpc.aio

from . import kowitodb_pb2 as pb
from . import kowitodb_pb2_grpc as pb_grpc

DEFAULT_TIMEOUT: Optional[float] = 30.0
"""Default per-RPC deadline (seconds) applied when a client is created without
an explicit ``timeout``."""


# ---- Channel construction: auth metadata + default deadline ----


class _CallDetails(
    collections.namedtuple(
        "_CallDetails",
        ("method", "timeout", "metadata", "credentials", "wait_for_ready", "compression"),
    ),
    grpc.ClientCallDetails,
):
    """Mutable-by-copy ``grpc.ClientCallDetails`` used by the sync interceptor."""


def _merge_metadata(existing, extra: Sequence[Tuple[str, str]]):
    """Return ``existing`` metadata plus ``extra`` (caller-supplied keys win)."""
    merged = list(existing or ())
    present = {k.lower() for k, _ in merged}
    merged.extend((k, v) for k, v in extra if k not in present)
    return merged


class _ClientInterceptor(
    grpc.UnaryUnaryClientInterceptor,
    grpc.UnaryStreamClientInterceptor,
    grpc.StreamUnaryClientInterceptor,
    grpc.StreamStreamClientInterceptor,
):
    """Adds the API key and a default deadline to every RPC (sync channel)."""

    def __init__(self, metadata: Sequence[Tuple[str, str]], timeout: Optional[float]):
        self._metadata = list(metadata)
        self._timeout = timeout

    def _details(self, d):
        return _CallDetails(
            d.method,
            d.timeout if d.timeout is not None else self._timeout,
            _merge_metadata(d.metadata, self._metadata),
            d.credentials,
            getattr(d, "wait_for_ready", None),
            getattr(d, "compression", None),
        )

    def intercept_unary_unary(self, continuation, client_call_details, request):
        return continuation(self._details(client_call_details), request)

    def intercept_unary_stream(self, continuation, client_call_details, request):
        return continuation(self._details(client_call_details), request)

    def intercept_stream_unary(self, continuation, client_call_details, request_iterator):
        return continuation(self._details(client_call_details), request_iterator)

    def intercept_stream_stream(self, continuation, client_call_details, request_iterator):
        return continuation(self._details(client_call_details), request_iterator)


class _AioClientInterceptor(
    grpc.aio.UnaryUnaryClientInterceptor,
    grpc.aio.UnaryStreamClientInterceptor,
):
    """Adds the API key and a default deadline to every RPC (asyncio channel)."""

    def __init__(self, metadata: Sequence[Tuple[str, str]], timeout: Optional[float]):
        self._metadata = list(metadata)
        self._timeout = timeout

    def _details(self, d):
        md = grpc.aio.Metadata(*_merge_metadata(d.metadata, self._metadata))
        return grpc.aio.ClientCallDetails(
            method=d.method,
            timeout=d.timeout if d.timeout is not None else self._timeout,
            metadata=md,
            credentials=d.credentials,
            wait_for_ready=d.wait_for_ready,
        )

    async def intercept_unary_unary(self, continuation, client_call_details, request):
        return await continuation(self._details(client_call_details), request)

    async def intercept_unary_stream(self, continuation, client_call_details, request):
        return await continuation(self._details(client_call_details), request)


def _auth_metadata(api_key: Optional[str]) -> List[Tuple[str, str]]:
    # The server accepts `authorization: Bearer <key>` (or `x-api-key: <key>`).
    return [("authorization", f"Bearer {api_key}")] if api_key else []


def _channel_credentials(
    secure: bool,
    root_certificates: Optional[bytes],
    credentials: Optional[grpc.ChannelCredentials],
) -> Optional[grpc.ChannelCredentials]:
    if credentials is not None:
        return credentials
    if secure or root_certificates is not None:
        return grpc.ssl_channel_credentials(root_certificates=root_certificates)
    return None


def _str_map(d: Optional[Dict[Any, Any]]) -> Dict[str, str]:
    """Metadata and metadata filters are string→string on the wire; stringify
    non-string values (e.g. ``{"year": 2024}`` → ``{"year": "2024"}``)."""
    return {str(k): v if isinstance(v, str) else str(v) for k, v in (d or {}).items()}


def _check_id(object_id: Optional[str]) -> Optional[str]:
    """Validate an optional caller-assigned object id (must be a UUID)."""
    if not object_id:
        return None
    try:
        uuid.UUID(str(object_id))
    except ValueError:
        raise ValueError(
            f"KowitoDB object ids must be UUIDs, got {object_id!r} "
            "(omit the id to let the server assign one)"
        ) from None
    return str(object_id)


def _insert_request(
    content: str,
    keywords=None,
    metadata=None,
    relationships=None,
    importance: float = 0.5,
    id: Optional[str] = None,
) -> "pb.InsertRequest":
    rels = [
        pb.RelationshipInput(relation_type=r[0], target_id=r[1])
        for r in (relationships or [])
    ]
    req = pb.InsertRequest(
        content=content,
        keywords=keywords or [],
        metadata=_str_map(metadata),
        relationships=rels,
        importance=importance,
    )
    object_id = _check_id(id)
    if object_id is not None:
        req.id = object_id
    return req


def _remember_request(
    content: str,
    keywords=None,
    metadata=None,
    importance: float = 0.5,
    id: Optional[str] = None,
) -> "pb.RememberRequest":
    req = pb.RememberRequest(
        content=content,
        keywords=keywords or [],
        metadata=_str_map(metadata),
        importance=importance,
    )
    object_id = _check_id(id)
    if object_id is not None:
        req.id = object_id
    return req


def _batch_insert_request(items: List[dict]) -> "pb.BatchInsertRequest":
    return pb.BatchInsertRequest(
        items=[
            _insert_request(
                item["content"],
                item.get("keywords"),
                item.get("metadata"),
                item.get("relationships"),
                item.get("importance", 0.5),
                item.get("id"),
            )
            for item in items
        ]
    )


# ---- Dataclasses with helpful reprs ----


def _truncate(s: str, n: int = 80) -> str:
    """Truncate a string for repr display."""
    if len(s) <= n:
        return s
    return s[:n] + "…"


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

    def __repr__(self) -> str:
        return (
            f"AskResult(id={self.id!r}, score={self.relevance_score:.3f}, "
            f"source={self.retrieval_source!r}, "
            f"content={_truncate(self.content)!r})"
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

    def __repr__(self) -> str:
        return (
            f"AskResponse(intent={self.detected_intent!r}, results={len(self.results)})"
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

    def __repr__(self) -> str:
        return f"SearchResult(id={self.id!r}, score={self.score:.3f})"


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

    def __repr__(self) -> str:
        return (
            f"Stats(objects={self.total_objects}, vectors={self.vector_count}, "
            f"graph=({self.graph_nodes}n/{self.graph_edges}e), "
            f"sessions={self.active_agent_sessions})"
        )


@dataclass
class UpdateResult:
    """Result of an update() call."""

    updated: bool
    version: int

    @classmethod
    def from_proto(cls, p: pb.UpdateResponse) -> "UpdateResult":
        return cls(updated=p.updated, version=p.version)

    def __repr__(self) -> str:
        return f"UpdateResult(updated={self.updated}, version={self.version})"


@dataclass
class KnowledgeObject:
    """A stored knowledge object (as returned by list())."""

    id: str
    content: str
    keywords: List[str] = field(default_factory=list)
    metadata: Dict[str, str] = field(default_factory=dict)
    importance: float = 0.0
    created_at: str = ""
    updated_at: str = ""

    @classmethod
    def from_proto(cls, p: pb.KnowledgeObject) -> "KnowledgeObject":
        return cls(
            id=p.id,
            content=p.content,
            keywords=list(p.keywords),
            metadata=dict(p.metadata),
            importance=p.importance,
            created_at=p.created_at,
            updated_at=p.updated_at,
        )

    def __repr__(self) -> str:
        return f"KnowledgeObject(id={self.id!r}, importance={self.importance:.2f})"


@dataclass
class ListResult:
    """Result of a list() call — a page of objects plus the total count."""

    objects: List[KnowledgeObject]
    total: int

    @classmethod
    def from_proto(cls, p: pb.ListResponse) -> "ListResult":
        return cls(
            objects=[KnowledgeObject.from_proto(o) for o in p.objects],
            total=p.total,
        )

    def __repr__(self) -> str:
        return f"ListResult(page={len(self.objects)}, total={self.total})"


@dataclass
class ConversationTurn:
    """A single turn in an agent conversation session."""

    role: str
    content: str
    timestamp: str = ""

    @classmethod
    def from_proto(cls, p: pb.ConversationTurnProto) -> "ConversationTurn":
        return cls(role=p.role, content=p.content, timestamp=p.timestamp)

    def __repr__(self) -> str:
        return (
            f"ConversationTurn(role={self.role!r}, content={_truncate(self.content)!r})"
        )


class KowitoDBClient:
    """Python gRPC client for KowitoDB.

    Usage:
        db = KowitoDBClient("localhost:50051")
        db.remember("Some knowledge to store")
        response = db.ask("What do you know about X?")

    Args:
        address: ``host:port`` of the KowitoDB gRPC server.
        api_key: if set, sent as ``authorization: Bearer <key>`` metadata on
            every RPC (matches the server's ``--api-key`` / ``KOWITODB_API_KEY``).
        timeout: default per-RPC deadline in seconds; ``None`` disables it.
        secure: use TLS (system root certificates unless ``root_certificates``
            is given). Implied by ``root_certificates`` or ``credentials``.
        root_certificates: PEM-encoded root certificates for TLS.
        credentials: explicit ``grpc.ChannelCredentials`` (overrides ``secure``).
        options: extra gRPC channel options, e.g.
            ``[("grpc.max_receive_message_length", 64 << 20)]``.
    """

    def __init__(
        self,
        address: str = "localhost:50051",
        *,
        api_key: Optional[str] = None,
        timeout: Optional[float] = DEFAULT_TIMEOUT,
        secure: bool = False,
        root_certificates: Optional[bytes] = None,
        credentials: Optional[grpc.ChannelCredentials] = None,
        options: Optional[Sequence[Tuple[str, Any]]] = None,
    ):
        self.address = address
        self.api_key = api_key
        self.timeout = timeout
        self._credentials = _channel_credentials(secure, root_certificates, credentials)
        self._options = list(options or [])
        self._raw_channel: Optional[grpc.Channel] = None
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
        """Create the gRPC channel.

        gRPC channels connect lazily: this does not contact the server. Call
        e.g. :meth:`stats` to verify the server is reachable.
        """
        if self._channel is not None:
            return
        if self._credentials is not None:
            raw = grpc.secure_channel(self.address, self._credentials, options=self._options)
        else:
            raw = grpc.insecure_channel(self.address, options=self._options)
        self._raw_channel = raw
        self._channel = grpc.intercept_channel(
            raw, _ClientInterceptor(_auth_metadata(self.api_key), self.timeout)
        )
        self._stub = pb_grpc.KowitoDBStub(self._channel)

    def close(self):
        """Close the gRPC connection."""
        if self._raw_channel is not None:
            self._raw_channel.close()
        self._raw_channel = None
        self._channel = None
        self._stub = None

    # ---- High-level AI API ----

    def ask(
        self,
        question: str,
        max_results: int = 10,
        metadata_filter: Optional[Dict[str, Any]] = None,
    ) -> AskResponse:
        """ai.ask() — natural-language query with automatic retrieval.

        The engine detects intent, chooses retrieval strategies,
        searches all indexes, reranks, and returns optimized results.

        ``metadata_filter`` applies exact-match metadata constraints (ANDed);
        empty or ``None`` means no filtering.
        """
        self._ensure_connected()
        req = pb.AskRequest(
            question=question,
            max_results=max_results,
            metadata_filter=_str_map(metadata_filter),
        )
        resp = self._stub.Ask(req)
        return AskResponse.from_proto(resp)

    def remember(
        self,
        content: str,
        keywords: Optional[List[str]] = None,
        metadata: Optional[Dict[str, str]] = None,
        importance: float = 0.5,
        id: Optional[str] = None,
    ) -> str:
        """ai.remember() — store knowledge for future retrieval.

        ``id`` optionally assigns the object id (must be a UUID string);
        otherwise the server generates one. Returns the object ID.
        """
        self._ensure_connected()
        req = _remember_request(content, keywords, metadata, importance, id)
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
        name to its string value. Columns: ``id``, ``content``, ``importance``,
        ``created_at``, ``updated_at``, ``keywords`` and ``metadata`` (the last
        two are JSON-encoded strings)::

            SELECT id, content FROM knowledge WHERE metadata LIKE '%"company":"Acme"%'
            SELECT content FROM knowledge WHERE keywords LIKE '%enterprise%' LIMIT 10
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
        id: Optional[str] = None,
    ) -> str:
        """Insert a knowledge object explicitly.

        ``id`` optionally assigns the object id (must be a UUID string);
        otherwise the server generates one. Returns the object ID.
        """
        self._ensure_connected()
        req = _insert_request(content, keywords, metadata, relationships, importance, id)
        resp = self._stub.Insert(req)
        return resp.id

    def batch_insert(self, items: List[dict]) -> List[str]:
        """Insert multiple knowledge objects in a single request.

        Each item is a dict mirroring :meth:`insert`:

            db.batch_insert([
                {"content": "...", "metadata": {...}, "keywords": [...],
                 "importance": 0.8},
                {"content": "..."},
            ])

        Supported keys per item: ``content`` (required), ``keywords``,
        ``metadata``, ``relationships`` (list of ``(relation_type, target_id)``
        tuples), ``importance``, and ``id`` (optional caller-assigned UUID).

        Returns the list of created object IDs, in input order.
        """
        self._ensure_connected()
        req = _batch_insert_request(items)
        resp = self._stub.BatchInsert(req)
        return list(resp.ids)

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
        req = pb.UpdateRequest(id=id, metadata=_str_map(metadata), keywords=keywords or [])
        if content is not None:
            req.content = content
        if importance is not None:
            req.importance = importance
        if change_description is not None:
            req.change_description = change_description
        resp = self._stub.Update(req)
        return UpdateResult.from_proto(resp)

    def search(
        self,
        query: str,
        top_k: int = 20,
        metadata_filter: Optional[Dict[str, Any]] = None,
    ) -> List[SearchResult]:
        """Direct search (bypasses the AI planner).

        ``metadata_filter`` applies exact-match metadata constraints (ANDed);
        empty or ``None`` means no filtering.
        """
        self._ensure_connected()
        req = pb.SearchRequest(
            query=query, top_k=top_k, metadata_filter=_str_map(metadata_filter)
        )
        resp = self._stub.Search(req)
        return [SearchResult.from_proto(r) for r in resp.results]

    def list(self, offset: int = 0, limit: int = 0) -> ListResult:
        """List stored knowledge objects (paginated scroll).

        ``offset`` is the number of objects to skip; ``limit`` is the page
        size, where ``0`` means the server default.

        Returns a ``ListResult`` exposing ``objects`` (a list of
        ``KnowledgeObject``) and ``total`` (the full object count).
        """
        self._ensure_connected()
        req = pb.ListRequest(offset=offset, limit=limit)
        resp = self._stub.List(req)
        return ListResult.from_proto(resp)

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


class AsyncKowitoDBClient:
    """Async Python gRPC client for KowitoDB.

    Use this with asyncio-based applications (FastAPI, LangChain async, etc.).
    Supports ``async with`` for automatic connection lifecycle management.

    Usage:
        async with AsyncKowitoDBClient("localhost:50051") as db:
            resp = await db.ask("What do you know about X?")

    Accepts the same keyword options as :class:`KowitoDBClient`
    (``api_key``, ``timeout``, ``secure``, ``root_certificates``,
    ``credentials``, ``options``).
    """

    def __init__(
        self,
        address: str = "localhost:50051",
        *,
        api_key: Optional[str] = None,
        timeout: Optional[float] = DEFAULT_TIMEOUT,
        secure: bool = False,
        root_certificates: Optional[bytes] = None,
        credentials: Optional[grpc.ChannelCredentials] = None,
        options: Optional[Sequence[Tuple[str, Any]]] = None,
    ):
        self.address = address
        self.api_key = api_key
        self.timeout = timeout
        self._credentials = _channel_credentials(secure, root_certificates, credentials)
        self._options = list(options or [])
        self._channel: Optional[grpc.aio.Channel] = None
        self._stub: Optional[pb_grpc.KowitoDBStub] = None

    # ---- Async context manager ----

    async def __aenter__(self):
        await self.connect()
        return self

    async def __aexit__(self, *args):
        await self.close()

    # ---- Connection ----

    async def connect(self):
        """Create the gRPC channel (connects lazily, on the first RPC)."""
        if self._channel is not None:
            return
        interceptors = [_AioClientInterceptor(_auth_metadata(self.api_key), self.timeout)]
        if self._credentials is not None:
            self._channel = grpc.aio.secure_channel(
                self.address, self._credentials, options=self._options,
                interceptors=interceptors,
            )
        else:
            self._channel = grpc.aio.insecure_channel(
                self.address, options=self._options, interceptors=interceptors
            )
        self._stub = pb_grpc.KowitoDBStub(self._channel)

    async def close(self):
        """Close the gRPC connection."""
        if self._channel is not None:
            await self._channel.close()
            self._channel = None
            self._stub = None

    # ---- High-level AI API ----

    async def ask(
        self,
        question: str,
        max_results: int = 10,
        metadata_filter: Optional[Dict[str, Any]] = None,
    ) -> AskResponse:
        """ai.ask() — natural-language query with automatic retrieval."""
        self._ensure_connected()
        req = pb.AskRequest(
            question=question,
            max_results=max_results,
            metadata_filter=_str_map(metadata_filter),
        )
        resp = await self._stub.Ask(req)
        return AskResponse.from_proto(resp)

    async def remember(
        self,
        content: str,
        keywords: Optional[List[str]] = None,
        metadata: Optional[Dict[str, str]] = None,
        importance: float = 0.5,
        id: Optional[str] = None,
    ) -> str:
        """ai.remember() — store knowledge for future retrieval."""
        self._ensure_connected()
        req = _remember_request(content, keywords, metadata, importance, id)
        resp = await self._stub.Remember(req)
        return resp.id

    async def forget(self, object_id: str) -> bool:
        """Remove a knowledge object by ID."""
        self._ensure_connected()
        req = pb.DeleteRequest(id=object_id)
        resp = await self._stub.Delete(req)
        return resp.existed

    # ---- SQL API ----

    async def sql(self, query: str) -> List[Dict[str, str]]:
        """Execute a SQL query against the DataFusion engine."""
        self._ensure_connected()
        req = pb.SqlRequest(query=query)
        resp = await self._stub.Sql(req)
        return [dict(row.columns) for row in resp.rows]

    # ---- Low-level API ----

    async def insert(
        self,
        content: str,
        keywords: Optional[List[str]] = None,
        metadata: Optional[Dict[str, str]] = None,
        relationships: Optional[List[tuple]] = None,
        importance: float = 0.5,
        id: Optional[str] = None,
    ) -> str:
        """Insert a knowledge object explicitly.

        ``id`` optionally assigns the object id (must be a UUID string);
        otherwise the server generates one. Returns the object ID.
        """
        self._ensure_connected()
        req = _insert_request(content, keywords, metadata, relationships, importance, id)
        resp = await self._stub.Insert(req)
        return resp.id

    async def batch_insert(self, items: List[dict]) -> List[str]:
        """Insert multiple knowledge objects in a single request."""
        self._ensure_connected()
        req = _batch_insert_request(items)
        resp = await self._stub.BatchInsert(req)
        return list(resp.ids)

    async def get(self, object_id: str) -> Optional[dict]:
        """Retrieve a knowledge object by ID."""
        self._ensure_connected()
        req = pb.GetRequest(id=object_id)
        resp = await self._stub.Get(req)
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

    async def update(
        self,
        id: str,
        content: Optional[str] = None,
        metadata: Optional[Dict[str, str]] = None,
        keywords: Optional[List[str]] = None,
        importance: Optional[float] = None,
        change_description: Optional[str] = None,
    ) -> UpdateResult:
        """Update an existing knowledge object."""
        self._ensure_connected()
        req = pb.UpdateRequest(id=id, metadata=_str_map(metadata), keywords=keywords or [])
        if content is not None:
            req.content = content
        if importance is not None:
            req.importance = importance
        if change_description is not None:
            req.change_description = change_description
        resp = await self._stub.Update(req)
        return UpdateResult.from_proto(resp)

    async def search(
        self,
        query: str,
        top_k: int = 20,
        metadata_filter: Optional[Dict[str, Any]] = None,
    ) -> List[SearchResult]:
        """Direct search (bypasses the AI planner)."""
        self._ensure_connected()
        req = pb.SearchRequest(
            query=query, top_k=top_k, metadata_filter=_str_map(metadata_filter)
        )
        resp = await self._stub.Search(req)
        return [SearchResult.from_proto(r) for r in resp.results]

    async def list(self, offset: int = 0, limit: int = 0) -> ListResult:
        """List stored knowledge objects (paginated scroll)."""
        self._ensure_connected()
        req = pb.ListRequest(offset=offset, limit=limit)
        resp = await self._stub.List(req)
        return ListResult.from_proto(resp)

    # ---- Agent conversation memory ----

    async def record_turn(self, session_id: str, role: str, content: str) -> int:
        """Record a conversation turn for an agent session."""
        self._ensure_connected()
        req = pb.RecordTurnRequest(session_id=session_id, role=role, content=content)
        resp = await self._stub.RecordTurn(req)
        return resp.turn_count

    async def get_session(self, session_id: str) -> Optional[List[ConversationTurn]]:
        """Retrieve all turns for an agent session."""
        self._ensure_connected()
        req = pb.GetSessionRequest(session_id=session_id)
        resp = await self._stub.GetSession(req)
        if not resp.found:
            return None
        return [ConversationTurn.from_proto(t) for t in resp.turns]

    async def stats(self) -> Stats:
        """Return database statistics."""
        self._ensure_connected()
        req = pb.StatsRequest()
        resp = await self._stub.Stats(req)
        return Stats.from_proto(resp)

    def _ensure_connected(self):
        if self._stub is None:
            raise RuntimeError(
                "Client is not connected. Use `await client.connect()` or "
                "`async with AsyncKowitoDBClient(...) as client:`."
            )
