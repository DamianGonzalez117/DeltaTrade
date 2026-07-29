"""Shared test fixtures for the DeltaTrade test suite."""
import json
from typing import Any, Self

import pytest

TEST_API_KEY = "test-api-key-1234"
TEST_API_SECRET = "test-api-secret-5678"


@pytest.fixture
def api_key() -> str:
    return TEST_API_KEY


@pytest.fixture
def api_secret() -> str:
    return TEST_API_SECRET





class FakeWebSocket:
    """Minimal async WebSocket double for BitsoWebSocketConnector tests.

    Mimics the subset of the ``websockets`` library API that the connector
    actually uses: ``send``, ``recv``, ``close`` and async-context-manager
    support via ``__aenter__`` / ``__aexit__``.
    """

    def __init__(self, incoming_messages: list[str] | None = None) -> None:
        self._incoming = list(incoming_messages or [])
        self._sent: list[str] = []
        self._closed = False
        self._connected = True

    async def __aenter__(self) -> Self:
        self._connected = True
        return self

    async def __aexit__(self, *exc: object) -> None:
        await self.close()

    async def send(self, message: str) -> None:
        self._sent.append(message)

    async def recv(self) -> str:
        if not self._incoming:
            await self.close()
            raise ConnectionError("No more messages")
        return self._incoming.pop(0)

    async def close(self) -> None:
        self._closed = True
        self._connected = False

    @property
    def sent_messages(self) -> list[str]:
        return list(self._sent)

    @property
    def sent_jsons(self) -> list[dict[str, Any]]:
        return [json.loads(m) for m in self._sent]

    @property
    def is_closed(self) -> bool:
        return self._closed


class FakeWebSocketFactory:
    """Factory that produces FakeWebSocket instances and records calls."""

    def __init__(self, ws: FakeWebSocket) -> None:
        self._ws = ws
        self.created = False

    async def __call__(self, uri: str) -> FakeWebSocket:
        self.created = True
        return self._ws


@pytest.fixture
def fake_ws():
    return FakeWebSocket


@pytest.fixture
def make_fake_ws_factory():
    def _factory(messages: list[str] | None = None) -> tuple[FakeWebSocket, FakeWebSocketFactory]:
        ws = FakeWebSocket(messages)
        return ws, FakeWebSocketFactory(ws)

    return _factory