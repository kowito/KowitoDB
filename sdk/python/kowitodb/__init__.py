"""
KowitoDB Python SDK

An AI Knowledge Operating System client — use `ai.ask()`, `ai.remember()`,
and `ai.forget()` instead of manually orchestrating vector databases,
search engines, and graph databases.
"""

from dataclasses import dataclass, field
from typing import Any, Dict, List, Optional

import grpc

# Proto stubs would be generated here. For now, we provide a clean API
# surface that matches the idea document's vision.


@dataclass
class AskResult:
    """A single result from an `ai.ask()` query."""

    id: str
    content: str
    relevance_score: float
    metadata: Dict[str, str] = field(default_factory=dict)
    retrieval_source: str = ""


@dataclass
class AskResponse:
    """Response from `ai.ask()`."""

    results: List[AskResult]
    plan_explanation: str
    detected_intent: str


class KowitoDBClient:
    """Python client for KowitoDB.

    Usage:
        db = KowitoDBClient("localhost:50051")
        db.remember("OpenAI raised $6.6B in 2024")
        results = db.ask("Which companies raised funding after 2023?")
        for r in results:
            print(f"{r.content} (score: {r.relevance_score:.2f})")
    """

    def __init__(self, address: str = "localhost:50051"):
        self.address = address
        self._channel = None
        self._stub = None

    def connect(self):
        """Establish the gRPC connection."""
        self._channel = grpc.insecure_channel(self.address)
        # In production, import generated stubs:
        # from kowitodb_pb2_grpc import KowitoDBStub
        # self._stub = KowitoDBStub(self._channel)

    def close(self):
        """Close the gRPC connection."""
        if self._channel:
            self._channel.close()

    def __enter__(self):
        self.connect()
        return self

    def __exit__(self, *args):
        self.close()

    # ---- High-level AI API ----

    def ask(self, question: str, max_results: int = 10) -> AskResponse:
        """The core API: ask a natural-language question.

        The engine automatically:
        - Detects intent (factoid, comparison, temporal, entity search, etc.)
        - Chooses retrieval strategies (vector, keyword, graph, metadata, time)
        - Merges and ranks results
        """
        # In production, this calls the gRPC Ask endpoint:
        # request = kowitodb_pb2.AskRequest(question=question, max_results=max_results)
        # response = self._stub.Ask(request)
        # return _convert_ask_response(response)

        raise NotImplementedError(
            "gRPC connection required. Install with: pip install kowitodb[grpc]"
        )

    def remember(
        self,
        content: str,
        metadata: Optional[Dict[str, str]] = None,
        keywords: Optional[List[str]] = None,
        importance: float = 0.5,
    ) -> str:
        """Store a knowledge object for future retrieval.

        Returns the ID of the stored object.
        """
        raise NotImplementedError("gRPC connection required.")

    def forget(self, object_id: str) -> bool:
        """Remove a knowledge object by ID."""
        raise NotImplementedError("gRPC connection required.")

    # ---- Low-level API (expert mode) ----

    def search_vectors(self, query: str, top_k: int = 20) -> List[AskResult]:
        """Direct vector search (bypasses the planner)."""
        raise NotImplementedError("gRPC connection required.")

    def search_keywords(self, query: str, top_k: int = 20) -> List[AskResult]:
        """Direct keyword search (bypasses the planner)."""
        raise NotImplementedError("gRPC connection required.")

    # ---- Management ----

    def stats(self) -> dict:
        """Return database statistics."""
        raise NotImplementedError("gRPC connection required.")
