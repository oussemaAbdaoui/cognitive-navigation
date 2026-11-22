"""Shared fixtures. The whole suite runs offline and without an API key.

* Real sockets are disabled, so any accidental network call fails loudly.
* LLM calls are served by ``httpx.MockTransport`` (see ``fake_llm``), so the
  real request-building and response-parsing code in
  ``LLMNavigationSystem._llm_plan_navigation`` is exercised.
"""

import json
import os
import socket

import httpx
import pytest

# Keep deepeval (imported by the benchmark modules) quiet and offline.
os.environ.setdefault("DEEPEVAL_TELEMETRY_OPT_OUT", "YES")
os.environ.setdefault("DEEPEVAL_DISABLE_DOTENV", "1")

FAKE_KEY = "test-key-not-a-real-key"


@pytest.fixture(autouse=True)
def no_network(monkeypatch):
    real_connect = socket.socket.connect
    loopback = ("127.0.0.1", "::1", "localhost")

    def guard(sock, address, *args, **kwargs):
        # asyncio on Windows builds its internal socketpair over loopback.
        if isinstance(address, tuple) and address[0] in loopback:
            return real_connect(sock, address, *args, **kwargs)
        raise RuntimeError(f"Network access is disabled in tests: {address!r}")

    monkeypatch.setattr(socket.socket, "connect", guard)
    monkeypatch.delenv("GROQ_API_KEY", raising=False)


@pytest.fixture
def make_nav():
    from src.core.navigation import LLMNavigationSystem

    def _make(target=(0, 5), kernel="native", **kwargs):
        return LLMNavigationSystem(
            target_position=target, kernel=kernel, api_key=FAKE_KEY, **kwargs
        )

    return _make


def llm_reply(action, **extra):
    """Build a chat-completions style body whose content wraps `action`."""
    content = json.dumps({"spatial_analysis": "test", "action": action, **extra})
    return {
        "choices": [{"message": {"content": f"```json\n{content}\n```"}}],
        "usage": {"total_tokens": 42},
    }


class FakeLLM:
    """Scripted replacement for the Groq endpoint.

    `replies` items are either a dict (JSON body, status 200) or an
    ``httpx.Response``. Every request is recorded in ``self.requests``.
    """

    def __init__(self):
        self.replies = []
        self.requests = []

    def handler(self, request: httpx.Request) -> httpx.Response:
        self.requests.append(request)
        if not self.replies:
            raise AssertionError("FakeLLM received more requests than scripted")
        reply = self.replies.pop(0)
        if isinstance(reply, httpx.Response):
            return reply
        return httpx.Response(200, json=reply)


@pytest.fixture
def fake_llm(monkeypatch):
    import src.core.navigation as navigation

    fake = FakeLLM()
    real_client = httpx.AsyncClient

    def client_factory(*args, **kwargs):
        kwargs["transport"] = httpx.MockTransport(fake.handler)
        return real_client(*args, **kwargs)

    monkeypatch.setattr(navigation.httpx, "AsyncClient", client_factory)
    return fake
