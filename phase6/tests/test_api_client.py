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

_CHAT_TURN_BODY = {
    "session_id": "s1", "run_id": "r1", "intent": "explain_plan",
    "assistant_message": "Galata Tower is a well-known landmark.", "response_language": "en",
    "trip_patch": None, "requires_new_run": False, "new_run_id": None, "new_run_status": None,
    "clarification_required": False, "warnings": [],
}


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


# --- Hybrid Chat C.1: create_chat_turn ----------------------------------------------------


def test_create_chat_turn_request_mapping(monkeypatch):
    captured = {}

    def _fake_post(path, json_body=None):
        captured["path"] = path
        captured["json_body"] = json_body
        return _FakeResponse(200, _CHAT_TURN_BODY)

    client = SystemAClient(base_url="http://agent-system-a:8010")
    monkeypatch.setattr(client, "_post", _fake_post)
    result = client.create_chat_turn("s1", "r1", "Why did you recommend Galata Tower?", "en")

    assert captured["path"] == "/v1/chat/turns"
    assert captured["json_body"] == {
        "session_id": "s1", "run_id": "r1",
        "user_message": "Why did you recommend Galata Tower?", "preferred_language": "en",
    }
    assert result.intent == "explain_plan"
    assert result.requires_new_run is False


def test_create_chat_turn_defaults_preferred_language_to_en(monkeypatch):
    captured = {}

    def _fake_post(path, json_body=None):
        captured["json_body"] = json_body
        return _FakeResponse(200, _CHAT_TURN_BODY)

    client = SystemAClient(base_url="http://agent-system-a:8010")
    monkeypatch.setattr(client, "_post", _fake_post)
    client.create_chat_turn("s1", "r1", "hi")
    assert captured["json_body"]["preferred_language"] == "en"


def test_create_chat_turn_parses_modify_trip_response_with_patch(monkeypatch):
    body = dict(_CHAT_TURN_BODY, intent="modify_trip", requires_new_run=True, new_run_id="r2",
                new_run_status="pending", trip_patch={"traveler_count": 3}, warnings=["unrecognized_interests_to_add:skydiving"])
    client = SystemAClient(base_url="http://agent-system-a:8010")
    monkeypatch.setattr(client, "_post", lambda path, json_body=None: _FakeResponse(200, body))
    result = client.create_chat_turn("s1", "r1", "add a traveler", "en")
    assert result.requires_new_run is True
    assert result.new_run_id == "r2"
    assert result.trip_patch == {"traveler_count": 3}
    assert result.warnings == ["unrecognized_interests_to_add:skydiving"]


def test_create_chat_turn_422_raises_response_error(monkeypatch):
    client = SystemAClient(base_url="http://agent-system-a:8010")
    monkeypatch.setattr(
        client, "_post",
        lambda path, json_body=None: _FakeResponse(422, {"schema_version": "1.0.0", "error_code": "CHAT_TURN_REJECTED", "message": "run_not_found_for_session", "trace_id": "x", "retriable": False}),
    )
    with pytest.raises(SystemAResponseError) as exc_info:
        client.create_chat_turn("s1", "no-such-run", "hi", "en")
    assert exc_info.value.error_code == "CHAT_TURN_REJECTED"


def test_create_chat_turn_missing_required_field_raises_response_error(monkeypatch):
    client = SystemAClient(base_url="http://agent-system-a:8010")
    incomplete = dict(_CHAT_TURN_BODY)
    del incomplete["assistant_message"]
    monkeypatch.setattr(client, "_post", lambda path, json_body=None: _FakeResponse(200, incomplete))
    with pytest.raises(SystemAResponseError):
        client.create_chat_turn("s1", "r1", "hi", "en")


# --- persistent-history/grounding correction: get_chat_history ---------------------------


def test_get_chat_history_request_mapping(monkeypatch):
    captured = {}

    def _fake_get(path):
        captured["path"] = path
        return _FakeResponse(200, {"schema_version": "1.0.0", "session_id": "s1", "run_id": "r1", "turns": [
            {"turn_id": "t1", "role": "user", "content": "hi", "intent": None, "response_language": None, "status": "completed", "created_at": "t"},
            {"turn_id": "t2", "role": "assistant", "content": "hello", "intent": "explain_plan", "response_language": "en", "status": "completed", "created_at": "t"},
        ]})

    client = SystemAClient(base_url="http://agent-system-a:8010")
    monkeypatch.setattr(client, "_get", _fake_get)
    turns = client.get_chat_history("s1", "r1")

    assert captured["path"] == "/v1/chat/sessions/s1/turns?run_id=r1"
    assert len(turns) == 2
    assert turns[0].role == "user" and turns[0].content == "hi"
    assert turns[1].intent == "explain_plan"


def test_get_chat_history_422_raises_response_error(monkeypatch):
    client = SystemAClient(base_url="http://agent-system-a:8010")
    monkeypatch.setattr(
        client, "_get",
        lambda path: _FakeResponse(422, {"schema_version": "1.0.0", "error_code": "CHAT_SESSION_ACCESS_REJECTED", "message": "denied", "trace_id": "x", "retriable": False}),
    )
    with pytest.raises(SystemAResponseError) as exc_info:
        client.get_chat_history("wrong-session", "r1")
    assert exc_info.value.error_code == "CHAT_SESSION_ACCESS_REJECTED"


def test_get_chat_history_missing_turns_field_raises_response_error(monkeypatch):
    client = SystemAClient(base_url="http://agent-system-a:8010")
    monkeypatch.setattr(client, "_get", lambda path: _FakeResponse(200, {"session_id": "s1", "run_id": "r1"}))
    with pytest.raises(SystemAResponseError):
        client.get_chat_history("s1", "r1")


def test_get_chat_history_empty_transcript_returns_empty_list(monkeypatch):
    client = SystemAClient(base_url="http://agent-system-a:8010")
    monkeypatch.setattr(client, "_get", lambda path: _FakeResponse(200, {"session_id": "s1", "run_id": "r1", "turns": []}))
    assert client.get_chat_history("s1", "r1") == []


# --- get_run: trip_request additive field -------------------------------------------------


def test_get_run_parses_trip_request_when_present(monkeypatch):
    client = SystemAClient(base_url="http://agent-system-a:8010")
    body = {
        "run_id": "r1", "session_id": "s1", "status": "completed", "created_at": "t", "updated_at": "t",
        "result": None, "trip_request": {"origin": "BEY", "destination": "IST"},
    }
    monkeypatch.setattr(client, "_get", lambda path: _FakeResponse(200, body))
    run = client.get_run("r1")
    assert run.trip_request == {"origin": "BEY", "destination": "IST"}


def test_get_run_trip_request_defaults_to_none_when_absent(monkeypatch):
    """Backward compatibility: an older server response with no
    trip_request field at all must still parse cleanly."""
    client = SystemAClient(base_url="http://agent-system-a:8010")
    body = {"run_id": "r1", "session_id": "s1", "status": "pending", "created_at": "t", "updated_at": "t"}
    monkeypatch.setattr(client, "_get", lambda path: _FakeResponse(200, body))
    run = client.get_run("r1")
    assert run.trip_request is None


# --- Hybrid Chat C.2: list_sessions --------------------------------------------------------


_SESSION_SUMMARY_BODY = {
    "session_id": "s1", "latest_run_id": "r1", "latest_run_status": "completed",
    "title": "BEY → Istanbul · 19–23 Sep", "origin": "BEY", "destination": "IST",
    "depart_date": "2026-09-19", "return_date": "2026-09-23", "preferred_language": "en",
    "chat_turn_count": 4, "created_at": "t1", "updated_at": "t2",
}


def test_list_sessions_request_mapping(monkeypatch):
    captured = {}

    def _fake_get(path):
        captured["path"] = path
        return _FakeResponse(200, {"sessions": [_SESSION_SUMMARY_BODY], "total": 1, "limit": 20, "offset": 0})

    client = SystemAClient(base_url="http://agent-system-a:8010")
    monkeypatch.setattr(client, "_get", _fake_get)
    summaries, total = client.list_sessions(limit=20, offset=0)

    assert captured["path"] == "/v1/chat/sessions?limit=20&offset=0"
    assert total == 1
    assert len(summaries) == 1
    assert summaries[0].title == "BEY → Istanbul · 19–23 Sep"
    assert summaries[0].chat_turn_count == 4


def test_list_sessions_default_limit_and_offset(monkeypatch):
    captured = {}

    def _fake_get(path):
        captured["path"] = path
        return _FakeResponse(200, {"sessions": [], "total": 0, "limit": 20, "offset": 0})

    client = SystemAClient(base_url="http://agent-system-a:8010")
    monkeypatch.setattr(client, "_get", _fake_get)
    client.list_sessions()
    assert captured["path"] == "/v1/chat/sessions?limit=20&offset=0"


def test_list_sessions_empty_list_returns_empty(monkeypatch):
    client = SystemAClient(base_url="http://agent-system-a:8010")
    monkeypatch.setattr(client, "_get", lambda path: _FakeResponse(200, {"sessions": [], "total": 0, "limit": 20, "offset": 0}))
    summaries, total = client.list_sessions()
    assert summaries == [] and total == 0


def test_list_sessions_missing_sessions_field_raises_response_error(monkeypatch):
    client = SystemAClient(base_url="http://agent-system-a:8010")
    monkeypatch.setattr(client, "_get", lambda path: _FakeResponse(200, {"total": 0}))
    with pytest.raises(SystemAResponseError):
        client.list_sessions()


def test_list_sessions_error_response_raises_response_error(monkeypatch):
    client = SystemAClient(base_url="http://agent-system-a:8010")
    monkeypatch.setattr(
        client, "_get",
        lambda path: _FakeResponse(500, {"schema_version": "1.0.0", "error_code": "INTERNAL", "message": "boom", "trace_id": "x", "retriable": False}),
    )
    with pytest.raises(SystemAResponseError):
        client.list_sessions()


def test_list_sessions_never_leaks_a_transcript_shaped_field(monkeypatch):
    """The parsed dataclass has no attribute that could hold a
    transcript/prompt/provider payload -- a structural guard against a
    future accidental leak."""
    client = SystemAClient(base_url="http://agent-system-a:8010")
    monkeypatch.setattr(client, "_get", lambda path: _FakeResponse(200, {"sessions": [_SESSION_SUMMARY_BODY], "total": 1, "limit": 20, "offset": 0}))
    summaries, _total = client.list_sessions()
    fields = vars(summaries[0]).keys()
    assert "turns" not in fields and "result" not in fields and "observations" not in fields


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
