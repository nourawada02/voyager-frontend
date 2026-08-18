"""Hermetic tests for `phase6.api_client.SystemAClient` -- every test uses
`httpx.MockTransport` (no real socket, no real agent-system-a process).
"""

from __future__ import annotations

import httpx
import pytest

from phase6.api_client import (
    SystemAClient,
    SystemAConfigurationError,
    SystemAConnectionError,
    SystemAResponseError,
    _parse_sse_lines,
)


class _FakeResponse:
    def __init__(self, status_code: int, json_body=None, text: str = ""):
        self.status_code = status_code
        self._json_body = json_body
        self.text = text

    def json(self):
        if self._json_body is None:
            raise ValueError("no json body")
        return self._json_body

    def read(self):
        return self.text.encode()


# --- configuration --------------------------------------------------------------------


def test_missing_base_url_raises_configuration_error(monkeypatch):
    monkeypatch.delenv("AGENT_SYSTEM_A_BASE_URL", raising=False)
    with pytest.raises(SystemAConfigurationError):
        SystemAClient()


def test_explicit_base_url_does_not_need_env_var(monkeypatch):
    monkeypatch.delenv("AGENT_SYSTEM_A_BASE_URL", raising=False)
    client = SystemAClient(base_url="http://agent-system-a:8010/")
    assert client.base_url == "http://agent-system-a:8010"  # trailing slash stripped


# --- health --------------------------------------------------------------------------


def test_health_parses_a_well_formed_response(monkeypatch):
    client = SystemAClient(base_url="http://agent-system-a:8010")
    monkeypatch.setattr(client, "_get", lambda path: _FakeResponse(200, {"status": "ok", "service": "agent-system-a", "mode": "fixture"}))
    result = client.health()
    assert result.status == "ok"
    assert result.mode == "fixture"


def test_health_missing_required_field_raises_response_error(monkeypatch):
    client = SystemAClient(base_url="http://agent-system-a:8010")
    monkeypatch.setattr(client, "_get", lambda path: _FakeResponse(200, {"status": "ok"}))  # missing "service"
    with pytest.raises(SystemAResponseError):
        client.health()


def test_health_non_200_raises_response_error_with_error_code(monkeypatch):
    client = SystemAClient(base_url="http://agent-system-a:8010")
    monkeypatch.setattr(
        client, "_get",
        lambda path: _FakeResponse(500, {"schema_version": "1.0.0", "error_code": "INTERNAL", "message": "boom", "trace_id": "x", "retriable": False}),
    )
    with pytest.raises(SystemAResponseError) as exc_info:
        client.health()
    assert exc_info.value.error_code == "INTERNAL"


def test_health_malformed_non_json_response_is_handled(monkeypatch):
    client = SystemAClient(base_url="http://agent-system-a:8010")

    class _BadJsonResponse:
        status_code = 502

        def json(self):
            raise ValueError("not json")

    monkeypatch.setattr(client, "_get", lambda path: _BadJsonResponse())
    with pytest.raises(SystemAResponseError):
        client.health()


def test_connection_error_is_wrapped(monkeypatch):
    client = SystemAClient(base_url="http://agent-system-a:8010")

    def _raise(path):
        raise SystemAConnectionError("Could not reach agent-system-a at http://agent-system-a:8010")

    monkeypatch.setattr(client, "_get", _raise)
    with pytest.raises(SystemAConnectionError):
        client.health()


# --- create_run ------------------------------------------------------------------------


def test_create_run_request_mapping(monkeypatch):
    captured = {}

    def _fake_post(path, json_body=None):
        captured["path"] = path
        captured["json_body"] = json_body
        return _FakeResponse(201, {"run_id": "r1", "session_id": "s1", "status": "pending"})

    client = SystemAClient(base_url="http://agent-system-a:8010")
    monkeypatch.setattr(client, "_post", _fake_post)
    result = client.create_run("Plan my trip", trip_request={"origin": "BEY"}, idempotency_key="abc")

    assert captured["path"] == "/v1/runs"
    assert captured["json_body"] == {"user_message": "Plan my trip", "trip_request": {"origin": "BEY"}, "idempotency_key": "abc"}
    assert result.run_id == "r1"


def test_create_run_omits_optional_fields_when_not_given(monkeypatch):
    captured = {}

    def _fake_post(path, json_body=None):
        captured["json_body"] = json_body
        return _FakeResponse(201, {"run_id": "r1", "session_id": "s1", "status": "pending"})

    client = SystemAClient(base_url="http://agent-system-a:8010")
    monkeypatch.setattr(client, "_post", _fake_post)
    client.create_run("hi")
    assert captured["json_body"] == {"user_message": "hi"}


def test_create_run_422_raises_response_error(monkeypatch):
    client = SystemAClient(base_url="http://agent-system-a:8010")
    monkeypatch.setattr(
        client, "_post",
        lambda path, json_body=None: _FakeResponse(422, {"schema_version": "1.0.0", "error_code": "REQUEST_VALIDATION_FAILED", "message": "bad field", "trace_id": "x", "retriable": False}),
    )
    with pytest.raises(SystemAResponseError) as exc_info:
        client.create_run("")
    assert exc_info.value.status_code == 422


# --- get_run / cancel_run ---------------------------------------------------------------


def test_get_run_parses_terminal_status_with_result(monkeypatch):
    client = SystemAClient(base_url="http://agent-system-a:8010")
    body = {
        "run_id": "r1", "session_id": "s1", "status": "completed",
        "created_at": "2026-08-18T00:00:00Z", "updated_at": "2026-08-18T00:01:00Z",
        "result": {"status": "success", "observations": [], "warnings": []},
    }
    monkeypatch.setattr(client, "_get", lambda path: _FakeResponse(200, body))
    run = client.get_run("r1")
    assert run.is_terminal is True
    assert run.result["status"] == "success"


def test_get_run_pending_is_not_terminal(monkeypatch):
    client = SystemAClient(base_url="http://agent-system-a:8010")
    body = {"run_id": "r1", "session_id": "s1", "status": "pending", "created_at": "t", "updated_at": "t", "result": None}
    monkeypatch.setattr(client, "_get", lambda path: _FakeResponse(200, body))
    run = client.get_run("r1")
    assert run.is_terminal is False


def test_get_run_404_raises_response_error(monkeypatch):
    client = SystemAClient(base_url="http://agent-system-a:8010")
    monkeypatch.setattr(
        client, "_get",
        lambda path: _FakeResponse(404, {"schema_version": "1.0.0", "error_code": "RUN_NOT_FOUND", "message": "not found", "trace_id": "x", "retriable": False}),
    )
    with pytest.raises(SystemAResponseError) as exc_info:
        client.get_run("unknown")
    assert exc_info.value.error_code == "RUN_NOT_FOUND"


def test_cancel_run_request_mapping(monkeypatch):
    captured = {}

    def _fake_post(path, json_body=None):
        captured["path"] = path
        return _FakeResponse(200, {"run_id": "r1", "session_id": "s1", "status": "running", "created_at": "t", "updated_at": "t", "result": None})

    client = SystemAClient(base_url="http://agent-system-a:8010")
    monkeypatch.setattr(client, "_post", _fake_post)
    result = client.cancel_run("r1")
    assert captured["path"] == "/v1/runs/r1/cancel"
    assert result.status == "running"


# --- SSE parsing -------------------------------------------------------------------------


def test_sse_parser_handles_a_full_ordered_stream():
    raw = (
        "id: e1\nevent: run_started\ndata: {\"run_id\": \"r1\"}\n\n"
        ": heartbeat\n\n"
        "id: e2\nevent: action_started\ndata: {\"action\": \"get_weather\"}\n\n"
        "id: e3\nevent: run_completed\ndata: {\"status\": \"success\"}\n\n"
    )
    lines = raw.split("\n")
    events = list(_parse_sse_lines(iter(lines)))
    assert [e.stage for e in events] == ["run_started", "action_started", "run_completed"]
    assert events[0].event_id == "e1"
    assert events[1].payload == {"action": "get_weather"}


def test_sse_parser_ignores_heartbeat_comment_lines():
    lines = [": heartbeat", "", ": heartbeat", ""]
    events = list(_parse_sse_lines(iter(lines)))
    assert events == []


def test_sse_parser_handles_malformed_data_gracefully():
    lines = ["id: e1", "event: run_started", "data: not-json{{{", ""]
    events = list(_parse_sse_lines(iter(lines)))
    assert len(events) == 1
    assert events[0].payload == {}  # malformed data never crashes the parser


def test_stream_events_supports_last_event_id_header(monkeypatch):
    captured = {}

    class _FakeStreamResponse:
        status_code = 200

        def iter_lines(self):
            return iter(["id: e2", "event: run_completed", "data: {}", ""])

        def __enter__(self):
            return self

        def __exit__(self, *a):
            return False

    def _fake_stream(method, url, headers=None, timeout=None):
        captured["headers"] = headers
        return _FakeStreamResponse()

    monkeypatch.setattr(httpx, "stream", _fake_stream)
    client = SystemAClient(base_url="http://agent-system-a:8010")
    events = list(client.stream_events("r1", last_event_id="e1"))
    assert captured["headers"]["Last-Event-ID"] == "e1"
    assert events[0].stage == "run_completed"


def test_stream_events_non_200_raises_response_error(monkeypatch):
    class _FakeErrorStreamResponse:
        status_code = 404
        text = ""

        def read(self):
            return b""

        def json(self):
            return {"schema_version": "1.0.0", "error_code": "RUN_NOT_FOUND", "message": "not found", "trace_id": "x", "retriable": False}

        def __enter__(self):
            return self

        def __exit__(self, *a):
            return False

    monkeypatch.setattr(httpx, "stream", lambda method, url, headers=None, timeout=None: _FakeErrorStreamResponse())
    client = SystemAClient(base_url="http://agent-system-a:8010")
    with pytest.raises(SystemAResponseError):
        list(client.stream_events("unknown"))


# --- no secret/prompt leakage in any exception text --------------------------------------


def test_error_messages_never_contain_a_bearer_token_or_api_key_shaped_string(monkeypatch):
    client = SystemAClient(base_url="http://agent-system-a:8010")
    monkeypatch.setattr(
        client, "_get",
        lambda path: _FakeResponse(500, {"schema_version": "1.0.0", "error_code": "INTERNAL", "message": "An internal error occurred.", "trace_id": "x", "retriable": False}),
    )
    with pytest.raises(SystemAResponseError) as exc_info:
        client.health()
    message = str(exc_info.value).lower()
    for forbidden in ("bearer", "api_key", "authorization", "password"):
        assert forbidden not in message
