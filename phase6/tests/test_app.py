"""Hermetic Streamlit tests for `phase6.app` using Streamlit's own
supported testing mechanism (`streamlit.testing.v1.AppTest`) -- no
browser framework, no real agent-system-a process. Every network call
`phase6.api_client` would make is intercepted by monkeypatching the
top-level `httpx.get`/`httpx.post`/`httpx.stream` functions (the exact
functions `SystemAClient` calls), which AppTest's same-process script
execution honors correctly.
"""

from __future__ import annotations

import json

import httpx
import pytest
from streamlit.testing.v1 import AppTest

APP_PATH = "phase6/app.py"


class _JsonResponse:
    def __init__(self, status_code: int, body: dict):
        self.status_code = status_code
        self._body = body
        self.text = json.dumps(body)

    def json(self):
        return self._body

    def read(self):
        return self.text.encode()


def _health_ok(mode: str = "fixture", run_status: str = "completed"):
    # `run_status` defaults to a TERMINAL status: AppTest actually drives
    # `st.rerun()` calls to completion within one `.run()` invocation
    # (verified directly -- a mock that kept reporting "pending" forever
    # made the app's own bounded polling loop spin until AppTest's own
    # timeout, since there was never a terminal event to stop on). Tests
    # that specifically want to observe in-progress behavior pass
    # run_status="running" explicitly and keep interactions short.
    result = {"status": "success", "observations": [], "warnings": []} if run_status == "completed" else None

    def _fake_get(url, timeout=None):
        if url.endswith("/health"):
            return _JsonResponse(200, {"status": "ok", "service": "agent-system-a", "mode": mode})
        if "/v1/runs/" in url:
            return _JsonResponse(200, {
                "run_id": "run-1", "session_id": "session-1", "status": run_status,
                "created_at": "t", "updated_at": "t", "result": result,
            })
        raise AssertionError(f"unexpected GET {url}")
    return _fake_get


@pytest.fixture(autouse=True)
def _base_url(monkeypatch):
    monkeypatch.setenv("AGENT_SYSTEM_A_BASE_URL", "http://agent-system-a:8010")


# --- app loads and shows the form ------------------------------------------------------


def test_app_loads_with_healthy_backend_and_shows_the_form(monkeypatch):
    monkeypatch.setattr(httpx, "get", _health_ok())
    at = AppTest.from_file(APP_PATH)
    at.run(timeout=15)

    assert not at.exception
    assert at.title[0].value == "🧭 VoyagerAI Istanbul"
    assert len(at.text_input) >= 1  # origin field
    assert len(at.button) >= 1  # form submit button


def test_fixture_mode_banner_is_shown_when_health_reports_fixture_mode(monkeypatch):
    monkeypatch.setattr(httpx, "get", _health_ok(mode="fixture"))
    at = AppTest.from_file(APP_PATH)
    at.run(timeout=15)
    assert not at.exception
    info_texts = " ".join(i.value for i in at.info)
    assert "fixture" in info_texts.lower() or "demo" in info_texts.lower()


def test_real_mode_does_not_show_the_fixture_banner(monkeypatch):
    monkeypatch.setattr(httpx, "get", _health_ok(mode="real"))
    at = AppTest.from_file(APP_PATH)
    at.run(timeout=15)
    assert not at.exception
    info_texts = " ".join(i.value for i in at.info)
    assert "fixture" not in info_texts.lower()


# --- unreachable backend is a clear error, never a crash ---------------------------------


def test_backend_unreachable_shows_a_clear_error_not_a_crash(monkeypatch):
    def _fake_get(url, timeout=None):
        raise httpx.ConnectError("connection refused")

    monkeypatch.setattr(httpx, "get", _fake_get)
    at = AppTest.from_file(APP_PATH)
    at.run(timeout=15)

    assert not at.exception  # the app itself must not crash
    assert len(at.error) >= 1
    error_text = " ".join(e.value for e in at.error).lower()
    assert "reach" in error_text or "cannot" in error_text


def test_missing_base_url_shows_a_clear_configuration_error(monkeypatch):
    monkeypatch.delenv("AGENT_SYSTEM_A_BASE_URL", raising=False)
    at = AppTest.from_file(APP_PATH)
    at.run(timeout=15)
    assert not at.exception
    assert len(at.error) >= 1


# --- form submission creates exactly one run --------------------------------------------


def test_submitting_the_form_creates_exactly_one_run(monkeypatch):
    monkeypatch.setattr(httpx, "get", _health_ok())
    captured_posts = []

    def _fake_post(url, json=None, timeout=None):
        captured_posts.append((url, json))
        return _JsonResponse(201, {"run_id": "run-1", "session_id": "session-1", "status": "pending"})

    monkeypatch.setattr(httpx, "post", _fake_post)

    def _fake_stream(method, url, headers=None, timeout=None):
        class _S:
            status_code = 200

            def iter_lines(self):
                return iter(["id: e1", "event: run_started", "data: {}", ""])

            def __enter__(self):
                return self

            def __exit__(self, *a):
                return False

        return _S()

    monkeypatch.setattr(httpx, "stream", _fake_stream)

    at = AppTest.from_file(APP_PATH)
    at.run(timeout=15)
    assert not at.exception

    # Fill the free-text field and submit via the form's submit button.
    at.text_area[0].set_value("Plan a relaxing trip.")
    at.button[0].click().run(timeout=15)

    assert not at.exception
    assert len(captured_posts) == 1
    url, body = captured_posts[0]
    assert url.endswith("/v1/runs")
    assert body["user_message"] == "Plan a relaxing trip."
    assert "idempotency_key" in body
    assert body["trip_request"]["destination"] == "IST"
    assert body["trip_request"]["origin"] == "BEY"


def test_double_click_does_not_submit_two_runs(monkeypatch):
    """The run_id-gated rendering means the form is no longer even present
    after the first successful submission -- this proves that
    structurally, not just by counting network calls after one click."""
    monkeypatch.setattr(httpx, "get", _health_ok())
    captured_posts = []

    def _fake_post(url, json=None, timeout=None):
        captured_posts.append((url, json))
        return _JsonResponse(201, {"run_id": "run-1", "session_id": "session-1", "status": "pending"})

    monkeypatch.setattr(httpx, "post", _fake_post)

    def _fake_stream(method, url, headers=None, timeout=None):
        class _S:
            status_code = 200

            def iter_lines(self):
                return iter([])

            def __enter__(self):
                return self

            def __exit__(self, *a):
                return False

        return _S()

    monkeypatch.setattr(httpx, "stream", _fake_stream)

    at = AppTest.from_file(APP_PATH)
    at.run(timeout=15)
    at.button[0].click().run(timeout=15)

    assert len(captured_posts) == 1
    # After the first submission, the form is gone -- no submit button
    # with the original form-submission semantics remains to re-click.
    assert not any(b.label == "Plan my trip" for b in at.button)


# --- invalid form input is rejected client-side, never sent -------------------------------


def test_return_date_before_depart_date_is_rejected_without_a_network_call(monkeypatch):
    from datetime import date, timedelta

    monkeypatch.setattr(httpx, "get", _health_ok())
    posts = []
    monkeypatch.setattr(httpx, "post", lambda *a, **k: posts.append(1))

    at = AppTest.from_file(APP_PATH)
    at.run(timeout=15)
    depart = date.today() + timedelta(days=30)
    at.date_input[0].set_value(depart)
    at.date_input[1].set_value(depart - timedelta(days=1))  # return before depart
    at.button[0].click().run(timeout=15)

    assert not at.exception
    assert not posts  # no create_run call was ever made


# --- named evidence: session-state survival, reconnect/replay, cancel, terminal states -----


def _fake_stream_single_event(stage: str, payload: dict | None = None):
    def _fake_stream(method, url, headers=None, timeout=None):
        class _S:
            status_code = 200

            def iter_lines(self):
                data = json.dumps(payload or {})
                return iter([f"id: e-{stage}", f"event: {stage}", f"data: {data}", ""])

            def __enter__(self):
                return self

            def __exit__(self, *a):
                return False

        return _S()

    return _fake_stream


def test_run_id_and_session_id_survive_a_streamlit_rerun(monkeypatch):
    """After a run is created, run_id/session_id must be the SAME value
    across multiple script reruns (Streamlit's own execution model reruns
    the whole script on every interaction) -- proves `st.session_state`
    persistence, not just a one-shot local variable.

    `st.rerun()` is patched to a no-op so this test can isolate ONE
    render pass at a time -- the app's own bottom-of-loop
    `time.sleep(...); st.rerun()` (unconditional whenever the run is not
    yet terminal) would otherwise make AppTest chase reruns until ITS
    own timeout, since this test's mock deliberately never reaches a
    terminal status (that "stops polling on a terminal status" property
    is covered separately, see
    `test_each_terminal_status_renders_correctly_and_stops_polling`)."""
    import streamlit

    monkeypatch.setattr(streamlit, "rerun", lambda: None)
    monkeypatch.setattr(httpx, "get", _health_ok(run_status="running"))
    monkeypatch.setattr(httpx, "post", lambda url, json=None, timeout=None: _JsonResponse(201, {"run_id": "run-xyz", "session_id": "session-xyz", "status": "pending"}))
    monkeypatch.setattr(httpx, "stream", _fake_stream_single_event("action_started", {"action": "get_weather"}))

    at = AppTest.from_file(APP_PATH)
    at.run(timeout=15)
    at.button[0].click().run(timeout=15)  # one pass: create the run, then one non-terminal render

    first_run_id = at.session_state["run_id"]
    first_session_id = at.session_state["session_id"]
    assert first_run_id == "run-xyz"
    assert first_session_id == "session-xyz"

    # A further rerun (e.g. the user interacting with something else, or
    # Streamlit's own internal rerun) must not regenerate or lose these
    # values -- proven by running the script again from the same
    # AppTest session and checking the values are unchanged.
    at.run(timeout=15)
    assert at.session_state["run_id"] == first_run_id
    assert at.session_state["session_id"] == first_session_id


def test_reconnect_with_last_event_id_never_duplicates_events_in_the_log(monkeypatch):
    """Simulates a reconnect: the first bounded read yields e1/e2, a
    second bounded read (as if the page reloaded and resumed from
    last_event_id=e2, exactly like the real server-side replay contract
    orchestration/system_a/run_store.py::get_event_sequence guarantees)
    yields only e3 -- the accumulated session-state event log must
    contain each event exactly once, in order, never a duplicate."""
    from phase6.app import _pull_new_events
    from phase6.api_client import SystemAClient

    calls = []

    def fake_stream_events(self, run_id, last_event_id=None):
        calls.append(last_event_id)
        from phase6.api_client import SSEEvent
        if last_event_id is None:
            yield SSEEvent(event_id="e1", stage="run_started", payload={})
            yield SSEEvent(event_id="e2", stage="action_started", payload={"action": "get_weather"})
        elif last_event_id == "e2":
            yield SSEEvent(event_id="e3", stage="run_completed", payload={"status": "success"})
        else:
            raise AssertionError(f"unexpected last_event_id {last_event_id}")

    monkeypatch.setattr(SystemAClient, "stream_events", fake_stream_events)
    client = SystemAClient(base_url="http://agent-system-a:8010")

    log: list[dict] = []
    last_id = None
    events1, last_id = _pull_new_events(client, "run-1", last_id)
    log.extend({"stage": e.stage, "payload": e.payload} for e in events1)
    events2, last_id = _pull_new_events(client, "run-1", last_id)
    log.extend({"stage": e.stage, "payload": e.payload} for e in events2)

    assert [e["stage"] for e in log] == ["run_started", "action_started", "run_completed"]
    assert len(log) == len(set(json.dumps(e, sort_keys=True) for e in log))  # no duplicates
    assert calls == [None, "e2"]  # the second read genuinely resumed from e2, never replayed e1/e2


def test_cancel_button_calls_the_cancel_endpoint(monkeypatch):
    """Same `st.rerun()` no-op technique as the session-survival test
    above -- isolates one non-terminal render pass so the "Cancel this
    run" button can actually be inspected and clicked, instead of the
    app's own polling loop chasing reruns forever against a mock that
    never reaches a terminal status."""
    import streamlit

    monkeypatch.setattr(streamlit, "rerun", lambda: None)
    monkeypatch.setattr(httpx, "get", _health_ok(run_status="running"))
    monkeypatch.setattr(httpx, "post", lambda url, json=None, timeout=None: _JsonResponse(201, {"run_id": "run-1", "session_id": "session-1", "status": "pending"}))
    monkeypatch.setattr(httpx, "stream", _fake_stream_single_event("action_started"))

    at = AppTest.from_file(APP_PATH)
    at.run(timeout=15)
    at.button[0].click().run(timeout=15)  # submit the form -> create_run + main()'s own (now no-op) rerun
    at.run(timeout=15)  # re-enter with run_id already set -> one non-terminal render of the run page

    cancel_buttons = [b for b in at.button if b.label == "Cancel this run"]
    assert cancel_buttons, "expected a visible Cancel this run button while the run is in progress"

    captured_cancel_calls = []

    def _fake_post_after_create(url, json=None, timeout=None):
        if url.endswith("/cancel"):
            captured_cancel_calls.append(url)
            return _JsonResponse(200, {"run_id": "run-1", "session_id": "session-1", "status": "running", "created_at": "t", "updated_at": "t", "result": None})
        raise AssertionError(f"unexpected POST {url}")

    monkeypatch.setattr(httpx, "post", _fake_post_after_create)
    cancel_buttons[0].click().run(timeout=15)

    assert any(c.endswith("/v1/runs/run-1/cancel") for c in captured_cancel_calls), captured_cancel_calls


@pytest.mark.parametrize("terminal_status,expected_marker", [
    ("completed", "Completed"),
    ("degraded", "Degraded"),
    ("failed", "Failed"),
    ("cancelled", "Cancelled"),
])
def test_each_terminal_status_renders_correctly_and_stops_polling(monkeypatch, terminal_status, expected_marker):
    """A single `.run()` call must be enough to reach the terminal render
    -- if the app kept polling (calling time.sleep + st.rerun again), a
    mock that never changes status would make AppTest's own internal
    rerun-loop run indefinitely and hit ITS timeout instead of returning
    promptly (this was observed directly and fixed earlier in this
    checkpoint's own test-authoring process)."""
    result = {
        "completed": {"status": "success", "observations": [], "warnings": []},
        "degraded": {"status": "partial", "observations": [], "warnings": ["some_capability_unavailable"]},
        "failed": {"status": "failed", "observations": [], "warnings": [], "reason": "internal_execution_error"},
        "cancelled": {"status": "degraded", "reason": "cancelled", "observations": [], "warnings": []},
    }[terminal_status]

    def _fake_get(url, timeout=None):
        if url.endswith("/health"):
            return _JsonResponse(200, {"status": "ok", "service": "agent-system-a", "mode": "fixture"})
        if "/v1/runs/" in url:
            return _JsonResponse(200, {
                "run_id": "run-1", "session_id": "session-1", "status": terminal_status,
                "created_at": "t", "updated_at": "t", "result": result,
            })
        raise AssertionError(f"unexpected GET {url}")

    monkeypatch.setattr(httpx, "get", _fake_get)
    monkeypatch.setattr(httpx, "post", lambda url, json=None, timeout=None: _JsonResponse(201, {"run_id": "run-1", "session_id": "session-1", "status": "pending"}))

    at = AppTest.from_file(APP_PATH)
    at.run(timeout=15)
    at.button[0].click().run(timeout=15)  # exactly one rerun reaches the terminal render

    assert not at.exception
    elements = list(at.success) + list(at.warning) + list(at.error)
    all_text = " ".join(x.value for x in elements if hasattr(x, "value"))
    assert expected_marker in all_text


# --- named evidence: bounded API timeouts ---------------------------------------------------


def test_all_client_timeouts_are_bounded_never_none():
    from phase6.api_client import SystemAClient

    client = SystemAClient(base_url="http://agent-system-a:8010")
    for attr in ("connect", "read", "write", "pool"):
        value = getattr(client._timeout, attr)
        assert value is not None and value > 0, f"timeout.{attr} must be a bounded positive number, got {value!r}"
    for attr in ("connect", "read", "write", "pool"):
        value = getattr(client._sse_timeout, attr)
        assert value is not None and value > 0, f"sse_timeout.{attr} must be a bounded positive number, got {value!r}"
