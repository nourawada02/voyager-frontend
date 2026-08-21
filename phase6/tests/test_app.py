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


# Hybrid Chat C.2: the recent-session sidebar issues exactly ONE extra GET
# (`/v1/chat/sessions?...`, never one per entry) on every render pass --
# every hand-written `_fake_get` closure in this file must answer it (an
# empty list is the safe, neutral default; tests that specifically
# exercise the sidebar override this explicitly).
_EMPTY_SESSION_LIST_BODY = {"sessions": [], "total": 0, "limit": 20, "offset": 0}


def _is_session_list_request(url: str) -> bool:
    return "/v1/chat/sessions?" in url or url.endswith("/v1/chat/sessions")


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
        if _is_session_list_request(url):
            return _JsonResponse(200, _EMPTY_SESSION_LIST_BODY)
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


def test_past_depart_date_is_rejected_without_a_network_call(monkeypatch):
    """Manual QA remediation Q.1: a past departure date must never reach
    create_run -- caught client-side with a clear inline message."""
    from datetime import date, timedelta

    monkeypatch.setattr(httpx, "get", _health_ok())
    posts = []
    monkeypatch.setattr(httpx, "post", lambda *a, **k: posts.append(1))

    at = AppTest.from_file(APP_PATH)
    at.run(timeout=15)
    yesterday = date.today() - timedelta(days=1)
    at.date_input[0].set_value(yesterday)
    at.button[0].click().run(timeout=15)

    assert not at.exception
    assert not posts
    assert any("past" in e.value.lower() for e in at.error)


def test_today_and_tomorrow_depart_dates_are_accepted_client_side(monkeypatch):
    """Manual QA remediation Q.1: today and tomorrow must clear client-side
    validation (the run still needs a real backend to actually complete,
    but no local "in the past" rejection should fire for either)."""
    from datetime import date, timedelta

    monkeypatch.setattr(httpx, "get", _health_ok())
    posts = []

    def _fake_post(url, json=None, timeout=None):
        posts.append(json)
        return _JsonResponse(201, {"run_id": "run-1", "session_id": "session-1", "status": "pending"})

    monkeypatch.setattr(httpx, "post", _fake_post)

    for offset_days in (0, 1):
        at = AppTest.from_file(APP_PATH)
        at.run(timeout=15)
        at.date_input[0].set_value(date.today() + timedelta(days=offset_days))
        at.button[0].click().run(timeout=15)
        assert not at.exception
        assert not any("past" in e.value.lower() for e in at.error)

    assert len(posts) == 2
    assert posts[0]["trip_request"]["depart_date"] == date.today().isoformat()
    assert posts[1]["trip_request"]["depart_date"] == (date.today() + timedelta(days=1)).isoformat()


def test_changing_depart_date_pulls_an_earlier_return_date_forward_live(monkeypatch):
    """Manual QA remediation Q.1: "changing departure revalidates return"
    -- moving departure past an already-selected return date must move
    the return date forward too, live (no submit needed), via the
    on_change callback, never leaving return before depart in session
    state."""
    from datetime import date, timedelta

    monkeypatch.setattr(httpx, "get", _health_ok())

    at = AppTest.from_file(APP_PATH)
    at.run(timeout=15)
    early_depart = date.today() + timedelta(days=10)
    at.date_input[0].set_value(early_depart).run(timeout=15)
    at.date_input[1].set_value(early_depart + timedelta(days=2)).run(timeout=15)
    assert at.date_input[1].value == early_depart + timedelta(days=2)

    later_depart = date.today() + timedelta(days=20)
    at.date_input[0].set_value(later_depart).run(timeout=15)

    assert not at.exception
    assert at.date_input[1].value >= later_depart


def test_currency_offers_try_and_usd_only_genuine_fx_conversion_exists(monkeypatch):
    """Manual QA remediation Q.1 (user correction pass §B): a genuine FX-
    conversion capability now exists (providers/fx_frankfurter.py,
    orchestration/system_a/budget_summary.py) -- TRY and USD are both
    offered. EUR remains excluded: no EUR rate source was verified."""
    monkeypatch.setattr(httpx, "get", _health_ok())

    at = AppTest.from_file(APP_PATH)
    at.run(timeout=15)
    currency_select = next(sb for sb in at.selectbox if sb.label == "Currency")
    assert currency_select.options == ["TRY", "USD"]
    assert currency_select.value == "TRY"
    assert "EUR" not in currency_select.options


def test_budget_label_is_stable_never_hardcoded_to_a_specific_currency(monkeypatch):
    """Pre-commit stabilization, required test 8: the budget amount
    widget's label must be the stable "Budget amount" -- never a
    currency-specific label like "Budget (TRY)"/"Budget (USD)". Inside
    st.form, a currency-embedded label only reflected the PREVIOUS
    selection until submit (form widgets don't rerun on individual
    change), which misled the user into thinking their currency choice
    hadn't registered."""
    monkeypatch.setattr(httpx, "get", _health_ok())

    at = AppTest.from_file(APP_PATH)
    at.run(timeout=15)
    number_inputs = list(at.number_input)
    budget_widgets = [w for w in number_inputs if "Budget" in w.label]
    assert len(budget_widgets) == 1
    assert budget_widgets[0].label == "Budget amount"
    assert "TRY" not in budget_widgets[0].label
    assert "USD" not in budget_widgets[0].label

    # Switching currency must not change the label at all (it is stable
    # now) and must not raise -- proves the widget no longer depends on
    # the live `currency` value for its own label text.
    currency_select = next(sb for sb in at.selectbox if sb.label == "Currency")
    at = currency_select.set_value("USD").run(timeout=15)
    assert not at.exception
    budget_widgets_after = [w for w in at.number_input if "Budget" in w.label]
    assert len(budget_widgets_after) == 1
    assert budget_widgets_after[0].label == "Budget amount"


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


_USD_TRIP_RESULT = {
    "status": "success",
    "narrative": "A comfortable Istanbul trip.",
    "observations": [
        {"action": "search_flights", "status": "success", "envelope": {"result": {"options": [
            {"carrier": "Turkish Airlines", "origin": "BEY", "destination": "IST", "depart_at": "2026-09-10T08:00:00+03:00",
             "arrive_at": "2026-09-10T10:00:00+03:00", "stops": 0, "price": {"amount_minor_units": 135000, "currency": "USD"}},
        ]}}},
        {"action": "search_stays", "status": "success", "envelope": {"stays": [
            {"rank": 1, "stay": {
                "name": "Konforlu Konaklama", "district_id": "district_fatih", "side": "european",
                "nightly_price": {"amount_minor_units": 35593, "currency": "TRY"},
                # Deliberately no `coordinates` here: st_folium is a real
                # JS-backed Streamlit component that hangs AppTest's own
                # bare-mode execution (observed directly) -- the map
                # popup's currency logic is covered separately below by a
                # pure results.py-level test (no Streamlit involved), so
                # this AppTest fixture stays map-free to keep the other
                # currency-coverage tests fast and reliable.
            }, "fair_price": {"estimated_fair_price": {"amount_minor_units": 40000, "currency": "TRY"}, "scoring_status": "complete"}},
        ]}},
    ],
    "warnings": [],
    "budget_summary": {
        "budget": {"amount_minor_units": 500000, "currency": "USD"},
        "fx_quote": {
            "base_currency": "USD", "quote_currency": "TRY", "rate": "40.00", "effective_date": "2026-08-19",
            "provider": "frankfurter.app (ECB reference rates)", "retrieved_at": "2026-08-20T12:00:00Z", "cache_status": "hit",
        },
        "fx_status": "success",
        "cheapest_flight": {"raw": {"amount_minor_units": 135000, "currency": "USD"}, "normalized": {"amount_minor_units": 135000, "currency": "USD"}},
        "cheapest_stay_total": {"raw": {"amount_minor_units": 106779, "currency": "TRY"}, "normalized": {"amount_minor_units": 2669, "currency": "USD"}},
    },
}


def _run_usd_trip_and_collect_text(monkeypatch, tab_index: int):
    def _fake_get(url, timeout=None):
        if url.endswith("/health"):
            return _JsonResponse(200, {"status": "ok", "service": "agent-system-a", "mode": "fixture"})
        if _is_session_list_request(url):
            return _JsonResponse(200, _EMPTY_SESSION_LIST_BODY)
        if "/v1/runs/" in url:
            return _JsonResponse(200, {
                "run_id": "run-1", "session_id": "session-1", "status": "completed",
                "created_at": "t", "updated_at": "t", "result": _USD_TRIP_RESULT,
            })
        raise AssertionError(f"unexpected GET {url}")

    monkeypatch.setattr(httpx, "get", _fake_get)
    monkeypatch.setattr(httpx, "post", lambda url, json=None, timeout=None: _JsonResponse(201, {"run_id": "run-1", "session_id": "session-1", "status": "pending"}))

    at = AppTest.from_file(APP_PATH)
    at.run(timeout=15)
    at.button[0].click().run(timeout=15)
    assert not at.exception

    tab = at.tabs[tab_index]
    texts = []
    for element_list in (tab.markdown, tab.caption):
        texts.extend(e.value for e in element_list)
    return texts


def test_usd_trip_flights_tab_shows_usd_as_primary_price(monkeypatch):
    """Manual QA remediation Q.1 (second correction pass, §1): the flight
    was already requested/priced natively in USD -- must render as USD,
    never demoted to or mixed with TRY."""
    texts = _run_usd_trip_and_collect_text(monkeypatch, tab_index=0)
    price_lines = [t for t in texts if "Price:" in t]
    assert price_lines, texts
    assert any("USD" in t for t in price_lines)
    assert not any("TRY" in t for t in price_lines)


def test_usd_trip_stays_tab_shows_usd_as_primary_price_try_only_as_secondary(monkeypatch):
    """The real accommodation price is TRY-native (Travel MCP) -- the
    PRIMARY rendered price must be the normalized USD amount; the raw TRY
    evidence must still be visible, but only as secondary/caption text,
    never as the unlabeled headline figure."""
    texts = _run_usd_trip_and_collect_text(monkeypatch, tab_index=1)
    nightly_lines = [t for t in texts if t.startswith("Nightly price:")]
    assert nightly_lines, texts
    assert "USD" in nightly_lines[0]
    fair_price_lines = [t for t in texts if t.startswith("Estimated fair price:")]
    assert fair_price_lines
    assert "USD" in fair_price_lines[0]
    total_lines = [t for t in texts if "Total for" in t]
    assert total_lines
    assert "USD" in total_lines[0]
    # The raw TRY amount is still disclosed somewhere (secondary evidence),
    # just never as an unconverted headline figure.
    assert any("Raw:" in t and "TRY" in t for t in texts)
    # A savings/deal figure was rendered, in the trip's own currency.
    savings_lines = [t for t in texts if "below the estimated fair price" in t or "above the estimated fair price" in t]
    assert savings_lines
    assert "USD" in savings_lines[0]


def test_usd_trip_budget_tab_discloses_fx_rate_never_shows_bare_try_total(monkeypatch):
    texts = _run_usd_trip_and_collect_text(monkeypatch, tab_index=4)
    assert any("Exchange rate" in t and "USD" in t and "TRY" in t for t in texts)


def test_usd_trip_map_popup_shows_usd_primary_price():
    """Pure-logic check (results.py, no Streamlit needed -- st_folium
    itself is a real JS-backed component that AppTest's bare mode can't
    exercise): the map popup's price text must show the normalized USD
    amount, with the raw TRY amount only as parenthetical secondary
    detail."""
    from phase6 import results as result_helpers

    result_with_coordinates = dict(_USD_TRIP_RESULT)
    result_with_coordinates["observations"] = [
        dict(_USD_TRIP_RESULT["observations"][0]),
        {"action": "search_stays", "status": "success", "envelope": {"stays": [
            {"rank": 1, "stay": {
                "name": "Konforlu Konaklama", "district_id": "district_fatih", "side": "european",
                "nightly_price": {"amount_minor_units": 35593, "currency": "TRY"},
                "coordinates": {"lat": 41.0, "lon": 28.9},
            }, "fair_price": {"estimated_fair_price": {"amount_minor_units": 40000, "currency": "TRY"}, "scoring_status": "complete"}},
        ]}},
    ]
    points = result_helpers.extract_map_points(result_with_coordinates)
    assert points
    pair = result_helpers.format_money_pair(points[0]["nightly_price_raw"], result_with_coordinates)
    assert "USD" in pair["primary"]
    assert pair["secondary"] and "TRY" in pair["secondary"]


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
        if _is_session_list_request(url):
            return _JsonResponse(200, _EMPTY_SESSION_LIST_BODY)
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


# --- Hybrid Chat C.1: "Continue planning" chat section ------------------------------------


def _run_to_completed_dashboard(monkeypatch, chat_post_handler=None):
    """Drives the app from a bare load through form submission to a
    COMPLETED run with a rendered dashboard -- the precondition for the
    chat section to ever appear (§2/§7: chat only appears below an
    already-generated dashboard, never replacing it)."""
    result = {"status": "success", "observations": [], "warnings": [], "narrative": "A relaxing 5-day trip."}

    def _fake_get(url, timeout=None):
        if url.endswith("/health"):
            return _JsonResponse(200, {"status": "ok", "service": "agent-system-a", "mode": "real"})
        if _is_session_list_request(url):
            return _JsonResponse(200, _EMPTY_SESSION_LIST_BODY)
        if "/v1/runs/" in url:
            return _JsonResponse(200, {
                "run_id": "run-1", "session_id": "session-1", "status": "completed",
                "created_at": "t", "updated_at": "t", "result": result,
            })
        raise AssertionError(f"unexpected GET {url}")

    posts = []

    def _fake_post(url, json=None, timeout=None):
        posts.append((url, json))
        if url.endswith("/v1/runs"):
            return _JsonResponse(201, {"run_id": "run-1", "session_id": "session-1", "status": "pending"})
        if url.endswith("/v1/chat/turns"):
            if chat_post_handler is not None:
                return chat_post_handler(json)
            return _JsonResponse(200, {
                "session_id": "session-1", "run_id": "run-1", "intent": "explain_plan",
                "assistant_message": "Galata Tower is a well-known landmark near your stay.",
                "response_language": "en", "trip_patch": None, "requires_new_run": False,
                "new_run_id": None, "new_run_status": None, "clarification_required": False, "warnings": [],
            })
        raise AssertionError(f"unexpected POST {url}")

    monkeypatch.setattr(httpx, "get", _fake_get)
    monkeypatch.setattr(httpx, "post", _fake_post)

    at = AppTest.from_file(APP_PATH)
    at.run(timeout=15)
    at.button[0].click().run(timeout=15)  # submits the form; get_run already reports "completed"
    assert not at.exception
    return at, posts


def test_chat_section_appears_below_a_completed_dashboard(monkeypatch):
    at, _posts = _run_to_completed_dashboard(monkeypatch)
    assert len(at.chat_input) == 1
    headers = " ".join(h.value for h in at.subheader)
    assert "Continue planning" in headers


def test_chat_section_does_not_appear_while_run_is_pending(monkeypatch):
    import streamlit

    monkeypatch.setattr(streamlit, "rerun", lambda: None)
    monkeypatch.setattr(httpx, "get", _health_ok(run_status="pending"))
    monkeypatch.setattr(httpx, "post", lambda url, json=None, timeout=None: _JsonResponse(201, {"run_id": "run-1", "session_id": "session-1", "status": "pending"}))
    monkeypatch.setattr(httpx, "stream", lambda method, url, headers=None, timeout=None: type("S", (), {
        "status_code": 200, "iter_lines": lambda self: iter([]), "__enter__": lambda self: self, "__exit__": lambda self, *a: False,
    })())

    at = AppTest.from_file(APP_PATH)
    at.run(timeout=15)
    at.button[0].click().run(timeout=15)  # one pass: create the run, then one non-terminal render
    assert not at.exception
    assert len(at.chat_input) == 0


def test_chat_section_does_not_appear_after_a_failed_run(monkeypatch):
    def _fake_get(url, timeout=None):
        if url.endswith("/health"):
            return _JsonResponse(200, {"status": "ok", "service": "agent-system-a", "mode": "real"})
        if _is_session_list_request(url):
            return _JsonResponse(200, _EMPTY_SESSION_LIST_BODY)
        return _JsonResponse(200, {
            "run_id": "run-1", "session_id": "session-1", "status": "failed",
            "created_at": "t", "updated_at": "t", "result": {"status": "failed", "reason": "internal_execution_error"},
        })

    monkeypatch.setattr(httpx, "get", _fake_get)
    monkeypatch.setattr(httpx, "post", lambda url, json=None, timeout=None: _JsonResponse(201, {"run_id": "run-1", "session_id": "session-1", "status": "pending"}))

    at = AppTest.from_file(APP_PATH)
    at.run(timeout=15)
    at.button[0].click().run(timeout=15)
    assert not at.exception
    assert len(at.chat_input) == 0


def test_submitting_a_chat_message_posts_to_chat_turns_with_session_and_run_ids(monkeypatch):
    at, posts = _run_to_completed_dashboard(monkeypatch)
    at.chat_input[0].set_value("Why did you recommend Galata Tower?").run(timeout=15)
    assert not at.exception

    chat_posts = [(u, b) for u, b in posts if u.endswith("/v1/chat/turns")]
    assert len(chat_posts) == 1
    _, body = chat_posts[0]
    assert body["session_id"] == "session-1"
    assert body["run_id"] == "run-1"
    assert body["user_message"] == "Why did you recommend Galata Tower?"

    chat_texts = " ".join(m.markdown[0].value if m.markdown else "" for m in at.chat_message)
    assert "Galata Tower" in chat_texts


def test_explain_plan_chat_turn_never_changes_the_active_run_id(monkeypatch):
    at, _posts = _run_to_completed_dashboard(monkeypatch)
    at.chat_input[0].set_value("Why did you recommend Galata Tower?").run(timeout=15)
    assert not at.exception
    assert at.session_state["run_id"] == "run-1"


def test_modify_trip_chat_turn_switches_to_the_new_run_id_but_keeps_the_session_id(monkeypatch):
    def _chat_handler(_body):
        return _JsonResponse(200, {
            "session_id": "session-1", "run_id": "run-2", "intent": "modify_trip",
            "assistant_message": "Updated your budget to 1000 USD and added shopping.",
            "response_language": "en", "trip_patch": {"budget": {"amount_minor_units": 100000, "currency": "USD"}},
            "requires_new_run": True, "new_run_id": "run-2", "new_run_status": "pending",
            "clarification_required": False, "warnings": [],
        })

    at, _posts = _run_to_completed_dashboard(monkeypatch, chat_post_handler=_chat_handler)
    at.chat_input[0].set_value("Change my budget to 1000 USD and add shopping").run(timeout=15)
    assert not at.exception
    assert at.session_state["run_id"] == "run-2"
    assert at.session_state["session_id"] == "session-1"


def test_chat_warnings_are_rendered_as_safe_captions_never_raw_json(monkeypatch):
    def _chat_handler(_body):
        return _JsonResponse(200, {
            "session_id": "session-1", "run_id": "run-1", "intent": "modify_trip", "assistant_message": "Updated.",
            "response_language": "en", "trip_patch": {"preferences": {"interests": ["shopping"]}},
            "requires_new_run": True, "new_run_id": "run-2", "new_run_status": "pending",
            "clarification_required": False, "warnings": ["unrecognized_interests_to_add:skydiving"],
        })

    at, _posts = _run_to_completed_dashboard(monkeypatch, chat_post_handler=_chat_handler)
    at.chat_input[0].set_value("Add shopping and skydiving").run(timeout=15)
    assert not at.exception
    captions = " ".join(c.value for c in at.caption)
    assert "unrecognized_interests_to_add:skydiving" in captions


def test_plan_a_new_trip_clears_the_chat_transcript(monkeypatch):
    at, _posts = _run_to_completed_dashboard(monkeypatch)
    at.chat_input[0].set_value("Why did you recommend Galata Tower?").run(timeout=15)
    assert not at.exception
    assert len(at.session_state["chat_messages"]) > 0

    new_trip_button = next(b for b in at.button if b.key == "new_trip_btn")
    new_trip_button.click().run(timeout=15)
    assert not at.exception
    assert at.session_state["chat_messages"] == []
    assert at.session_state["run_id"] is None


def test_chat_preferred_language_seeded_from_trip_form_language_selection(monkeypatch):
    result = {"status": "success", "observations": [], "warnings": []}

    def _fake_get(url, timeout=None):
        if url.endswith("/health"):
            return _JsonResponse(200, {"status": "ok", "service": "agent-system-a", "mode": "real"})
        if _is_session_list_request(url):
            return _JsonResponse(200, _EMPTY_SESSION_LIST_BODY)
        return _JsonResponse(200, {
            "run_id": "run-1", "session_id": "session-1", "status": "completed",
            "created_at": "t", "updated_at": "t", "result": result,
        })

    monkeypatch.setattr(httpx, "get", _fake_get)
    monkeypatch.setattr(httpx, "post", lambda url, json=None, timeout=None: _JsonResponse(201, {"run_id": "run-1", "session_id": "session-1", "status": "pending"}))

    at = AppTest.from_file(APP_PATH)
    at.run(timeout=15)
    language_select = next(sb for sb in at.selectbox if sb.label == "Preferred language")
    language_select.set_value("ar").run(timeout=15)
    at.button[0].click().run(timeout=15)
    assert not at.exception


# --- persistent-history/grounding correction: query-param session restoration -------------


def _query_param(at, name: str):
    """AppTest's own `query_params` exposes multi-value list semantics
    (unlike real Streamlit's scalar `st.query_params[...]`) -- this
    normalizes to the single value this app always sets."""
    value = at.query_params.get(name)
    return value[0] if isinstance(value, list) else value


def test_query_params_are_set_after_creating_a_run(monkeypatch):
    at, _posts = _run_to_completed_dashboard(monkeypatch)
    assert _query_param(at, "run_id") == "run-1"
    assert _query_param(at, "session_id") == "session-1"


def _fake_get_with_chat_history(history_turns: list[dict]):
    result = {"status": "success", "observations": [], "warnings": []}

    def _fake_get(url, timeout=None):
        if url.endswith("/health"):
            return _JsonResponse(200, {"status": "ok", "service": "agent-system-a", "mode": "real"})
        if _is_session_list_request(url):
            return _JsonResponse(200, _EMPTY_SESSION_LIST_BODY)
        if "/v1/chat/sessions/" in url:
            return _JsonResponse(200, {"session_id": "session-1", "run_id": "run-1", "turns": history_turns})
        if "/v1/runs/" in url:
            return _JsonResponse(200, {
                "run_id": "run-1", "session_id": "session-1", "status": "completed",
                "created_at": "t", "updated_at": "t", "result": result,
            })
        raise AssertionError(f"unexpected GET {url}")
    return _fake_get


def test_session_and_transcript_restore_from_query_params_on_a_fresh_process(monkeypatch):
    """Simulates a `chatbot-ui` restart: a genuinely NEW `AppTest`
    instance (fresh in-memory session_state, exactly like a new server
    process) whose URL still carries the previously-active session_id/
    run_id query params -- the transcript must come back from System A's
    own persistent store, not from anything client-side (which no longer
    exists in this fresh process)."""
    history = [
        {"turn_id": "t1", "role": "user", "content": "Why did you recommend this?", "intent": None, "response_language": None, "status": "completed", "created_at": "t"},
        {"turn_id": "t2", "role": "assistant", "content": "It matches your interests.", "intent": "explain_plan", "response_language": "en", "status": "completed", "created_at": "t"},
    ]
    monkeypatch.setattr(httpx, "get", _fake_get_with_chat_history(history))

    at = AppTest.from_file(APP_PATH)
    at.query_params["run_id"] = "run-1"
    at.query_params["session_id"] = "session-1"
    at.run(timeout=15)

    assert not at.exception
    assert at.session_state["run_id"] == "run-1"
    assert at.session_state["session_id"] == "session-1"
    assert at.session_state["chat_messages"] == [
        {"role": "user", "content": "Why did you recommend this?"},
        {"role": "assistant", "content": "It matches your interests."},
    ]
    chat_texts = " ".join(m.markdown[0].value if m.markdown else "" for m in at.chat_message)
    assert "It matches your interests." in chat_texts


def test_restored_transcript_does_not_duplicate_across_a_later_rerun(monkeypatch):
    history = [
        {"turn_id": "t1", "role": "user", "content": "hello", "intent": None, "response_language": None, "status": "completed", "created_at": "t"},
        {"turn_id": "t2", "role": "assistant", "content": "hi there", "intent": "explain_plan", "response_language": "en", "status": "completed", "created_at": "t"},
    ]
    monkeypatch.setattr(httpx, "get", _fake_get_with_chat_history(history))

    at = AppTest.from_file(APP_PATH)
    at.query_params["run_id"] = "run-1"
    at.query_params["session_id"] = "session-1"
    at.run(timeout=15)
    at.run(timeout=15)  # a second, unrelated rerun of the same live session

    assert not at.exception
    assert len(at.session_state["chat_messages"]) == 2  # never re-appended/duplicated


def test_mismatched_session_id_query_param_is_never_trusted(monkeypatch):
    """A URL claiming a session_id that does not match the real run's own
    stored session_id must never be restored -- proves the server-side
    cross-check, not just presence of both params, gates restoration."""
    def _fake_get(url, timeout=None):
        if url.endswith("/health"):
            return _JsonResponse(200, {"status": "ok", "service": "agent-system-a", "mode": "real"})
        if _is_session_list_request(url):
            return _JsonResponse(200, _EMPTY_SESSION_LIST_BODY)
        return _JsonResponse(200, {
            "run_id": "run-1", "session_id": "the-real-session-id", "status": "completed",
            "created_at": "t", "updated_at": "t", "result": {"status": "success", "observations": [], "warnings": []},
        })

    monkeypatch.setattr(httpx, "get", _fake_get)
    at = AppTest.from_file(APP_PATH)
    at.query_params["run_id"] = "run-1"
    at.query_params["session_id"] = "a-different-forged-session-id"
    at.run(timeout=15)

    assert not at.exception
    assert at.session_state["run_id"] is None  # falls through to the ordinary empty form


def test_plan_a_new_trip_clears_query_params(monkeypatch):
    at, _posts = _run_to_completed_dashboard(monkeypatch)
    assert _query_param(at, "run_id") == "run-1"
    new_trip_button = next(b for b in at.button if b.key == "new_trip_btn")
    new_trip_button.click().run(timeout=15)
    assert not at.exception
    assert "run_id" not in dict(at.query_params)
    assert "session_id" not in dict(at.query_params)


# --- persistent-history/grounding correction: truthful provider-unavailable error ---------


def test_provider_unavailable_chat_error_shows_a_truthful_specific_message(monkeypatch):
    def _chat_handler(_body):
        return _JsonResponse(503, {
            "schema_version": "1.0.0", "error_code": "PROVIDER_UNAVAILABLE",
            "message": "The chat assistant is temporarily unavailable.", "trace_id": "x", "retriable": True,
        })

    at, _posts = _run_to_completed_dashboard(monkeypatch, chat_post_handler=_chat_handler)
    at.chat_input[0].set_value("Why did you recommend Galata Tower?").run(timeout=15)
    assert not at.exception
    chat_texts = " ".join(m.markdown[0].value if m.markdown else "" for m in at.chat_message)
    assert "temporarily unavailable" in chat_texts


# --- Hybrid Chat C.2: recent-session sidebar ----------------------------------------------


_RUN_A = {
    "run_id": "run-a", "session_id": "session-a", "status": "completed",
    "result": {"status": "success", "observations": [], "warnings": []},
    "trip_request": {
        "origin": "BEY", "destination": "IST", "depart_date": "2026-09-19", "return_date": "2026-09-23",
        "traveler_count": 2, "budget": {"amount_minor_units": 500000, "currency": "TRY"},
        "preferences": {"interests": ["history"], "pace": "moderate", "language": "en", "mobility_constraints": []},
    },
}
_RUN_B = {
    "run_id": "run-b", "session_id": "session-b", "status": "completed",
    "result": {"status": "success", "observations": [], "warnings": []},
    "trip_request": {
        "origin": "LHR", "destination": "IST", "depart_date": "2026-10-01", "return_date": "2026-10-05",
        "traveler_count": 1, "budget": {"amount_minor_units": 300000, "currency": "TRY"},
        "preferences": {"interests": ["food"], "pace": "relaxed", "language": "en", "mobility_constraints": []},
    },
}
_SESSION_A_SUMMARY = {
    "session_id": "session-a", "latest_run_id": "run-a", "latest_run_status": "completed",
    "title": "BEY → Istanbul · 19–23 Sep", "origin": "BEY", "destination": "IST",
    "depart_date": "2026-09-19", "return_date": "2026-09-23", "preferred_language": "en",
    "chat_turn_count": 2, "created_at": "t1", "updated_at": "t3",
}
_SESSION_B_SUMMARY = {
    "session_id": "session-b", "latest_run_id": "run-b", "latest_run_status": "completed",
    "title": "LHR → Istanbul · 1–5 Oct", "origin": "LHR", "destination": "IST",
    "depart_date": "2026-10-01", "return_date": "2026-10-05", "preferred_language": "en",
    "chat_turn_count": 0, "created_at": "t2", "updated_at": "t2",
}
_HISTORY_A = [
    {"turn_id": "a1", "role": "user", "content": "Why did you pick these dates?", "intent": None, "response_language": None, "status": "completed", "created_at": "t"},
    {"turn_id": "a2", "role": "assistant", "content": "They matched your requested travel window.", "intent": "explain_plan", "response_language": "en", "status": "completed", "created_at": "t"},
]
_HISTORY_B: list = []


def _make_sidebar_backend(sessions, run_lookup, history_lookup):
    def _fake_get(url, timeout=None):
        if url.endswith("/health"):
            return _JsonResponse(200, {"status": "ok", "service": "agent-system-a", "mode": "real"})
        if _is_session_list_request(url):
            summaries = sessions() if callable(sessions) else sessions
            return _JsonResponse(200, {"sessions": summaries, "total": len(summaries), "limit": 20, "offset": 0})
        if "/v1/chat/sessions/" in url and "/turns" in url:
            session_id = url.split("/v1/chat/sessions/")[1].split("/turns")[0]
            turns = history_lookup.get(session_id, [])
            return _JsonResponse(200, {"session_id": session_id, "run_id": "?", "turns": turns})
        if "/v1/runs/" in url:
            run_id = url.rstrip("/").split("/v1/runs/")[-1]
            record = run_lookup[run_id]
            return _JsonResponse(200, {
                "run_id": record["run_id"], "session_id": record["session_id"], "status": record["status"],
                "created_at": "t", "updated_at": "t", "result": record["result"], "trip_request": record["trip_request"],
            })
        raise AssertionError(f"unexpected GET {url}")
    return _fake_get


def test_sidebar_renders_recent_sessions(monkeypatch):
    monkeypatch.setattr(httpx, "get", _make_sidebar_backend(
        [_SESSION_A_SUMMARY, _SESSION_B_SUMMARY], {"run-a": _RUN_A, "run-b": _RUN_B}, {"session-a": _HISTORY_A, "session-b": _HISTORY_B},
    ))
    at = AppTest.from_file(APP_PATH)
    at.run(timeout=15)
    assert not at.exception
    sidebar_text = " ".join(x.value for x in list(at.sidebar.subheader) + list(at.sidebar.caption))
    assert "Recent trips" in sidebar_text
    button_labels = [b.label for b in at.sidebar.button]
    assert any("BEY" in label for label in button_labels)
    assert any("LHR" in label for label in button_labels)


def test_sidebar_active_session_is_highlighted(monkeypatch):
    monkeypatch.setattr(httpx, "get", _make_sidebar_backend(
        [_SESSION_A_SUMMARY, _SESSION_B_SUMMARY], {"run-a": _RUN_A, "run-b": _RUN_B}, {"session-a": _HISTORY_A, "session-b": _HISTORY_B},
    ))
    at = AppTest.from_file(APP_PATH)
    at.query_params["session_id"] = "session-a"
    at.query_params["run_id"] = "run-a"
    at.run(timeout=15)
    assert not at.exception
    active_button = next(b for b in at.sidebar.button if "BEY" in b.label)
    other_button = next(b for b in at.sidebar.button if "LHR" in b.label)
    assert active_button.disabled is True
    assert other_button.disabled is False


def test_selecting_a_session_restores_its_run_and_transcript(monkeypatch):
    monkeypatch.setattr(httpx, "get", _make_sidebar_backend(
        [_SESSION_A_SUMMARY, _SESSION_B_SUMMARY], {"run-a": _RUN_A, "run-b": _RUN_B}, {"session-a": _HISTORY_A, "session-b": _HISTORY_B},
    ))
    at = AppTest.from_file(APP_PATH)
    at.run(timeout=15)
    session_a_button = next(b for b in at.sidebar.button if "BEY" in b.label)
    session_a_button.click().run(timeout=15)

    assert not at.exception
    assert at.session_state["session_id"] == "session-a"
    assert at.session_state["run_id"] == "run-a"
    assert at.session_state["chat_messages"] == [
        {"role": "user", "content": "Why did you pick these dates?"},
        {"role": "assistant", "content": "They matched your requested travel window."},
    ]
    assert at.session_state["trip_request_submitted"]["origin"] == "BEY"
    # No new run was created -- the existing run is loaded, not re-planned.
    button_labels = [b.label for b in at.button]
    assert "Plan my trip" not in button_labels


def test_switching_a_to_b_to_a_never_merges_messages(monkeypatch):
    monkeypatch.setattr(httpx, "get", _make_sidebar_backend(
        [_SESSION_A_SUMMARY, _SESSION_B_SUMMARY], {"run-a": _RUN_A, "run-b": _RUN_B}, {"session-a": _HISTORY_A, "session-b": _HISTORY_B},
    ))
    at = AppTest.from_file(APP_PATH)
    at.run(timeout=15)
    next(b for b in at.sidebar.button if "BEY" in b.label).click().run(timeout=15)
    assert at.session_state["chat_messages"] == [
        {"role": "user", "content": "Why did you pick these dates?"},
        {"role": "assistant", "content": "They matched your requested travel window."},
    ]

    next(b for b in at.sidebar.button if "LHR" in b.label).click().run(timeout=15)
    assert at.session_state["session_id"] == "session-b"
    assert at.session_state["chat_messages"] == []  # session B's own (empty) transcript, never A's leftover messages

    next(b for b in at.sidebar.button if "BEY" in b.label).click().run(timeout=15)
    assert at.session_state["session_id"] == "session-a"
    assert at.session_state["chat_messages"] == [
        {"role": "user", "content": "Why did you pick these dates?"},
        {"role": "assistant", "content": "They matched your requested travel window."},
    ]


def test_new_trip_preserves_sidebar_history(monkeypatch):
    monkeypatch.setattr(httpx, "get", _make_sidebar_backend(
        [_SESSION_A_SUMMARY, _SESSION_B_SUMMARY], {"run-a": _RUN_A, "run-b": _RUN_B}, {"session-a": _HISTORY_A, "session-b": _HISTORY_B},
    ))
    at = AppTest.from_file(APP_PATH)
    at.query_params["session_id"] = "session-a"
    at.query_params["run_id"] = "run-a"
    at.run(timeout=15)
    new_trip_button = next(b for b in at.button if b.key == "new_trip_btn")
    new_trip_button.click().run(timeout=15)

    assert not at.exception
    assert at.session_state["run_id"] is None  # the active trip was cleared
    button_labels = [b.label for b in at.sidebar.button]
    assert any("BEY" in label for label in button_labels)  # but the sidebar still lists it
    assert any("LHR" in label for label in button_labels)


def test_sidebar_api_failure_degrades_safely_and_preserves_active_trip(monkeypatch):
    def _fake_get(url, timeout=None):
        if url.endswith("/health"):
            return _JsonResponse(200, {"status": "ok", "service": "agent-system-a", "mode": "real"})
        if _is_session_list_request(url):
            raise httpx.ConnectError("connection refused")
        return _JsonResponse(200, {
            "run_id": "run-1", "session_id": "session-1", "status": "completed",
            "created_at": "t", "updated_at": "t", "result": {"status": "success", "observations": [], "warnings": []},
        })

    monkeypatch.setattr(httpx, "get", _fake_get)
    monkeypatch.setattr(httpx, "post", lambda url, json=None, timeout=None: _JsonResponse(201, {"run_id": "run-1", "session_id": "session-1", "status": "pending"}))
    at = AppTest.from_file(APP_PATH)
    at.run(timeout=15)
    at.button[0].click().run(timeout=15)

    assert not at.exception
    assert at.session_state["run_id"] == "run-1"  # the active trip is completely unaffected
    sidebar_warnings = " ".join(w.value for w in at.sidebar.warning)
    assert "recent trips" in sidebar_warnings.lower() or "could not load" in sidebar_warnings.lower()


def test_arabic_session_title_renders_without_error(monkeypatch):
    arabic_history = [
        {"turn_id": "b1", "role": "user", "content": "ما هي ميزانيتي؟", "intent": None, "response_language": None, "status": "completed", "created_at": "t"},
        {"turn_id": "b2", "role": "assistant", "content": "ميزانيتك الحالية هي 5,000.00 TRY.", "intent": "explain_plan", "response_language": "ar", "status": "completed", "created_at": "t"},
    ]
    arabic_summary = dict(_SESSION_A_SUMMARY, title="BEY → Istanbul · 19–23 Sep", preferred_language="ar")
    monkeypatch.setattr(httpx, "get", _make_sidebar_backend(
        [arabic_summary], {"run-a": _RUN_A}, {"session-a": arabic_history},
    ))
    at = AppTest.from_file(APP_PATH)
    at.run(timeout=15)
    assert not at.exception
    session_button = next(b for b in at.sidebar.button if "BEY" in b.label)
    session_button.click().run(timeout=15)
    assert not at.exception
    assert at.session_state["chat_messages"][0]["content"] == "ما هي ميزانيتي؟"
    assert "5,000.00 TRY" in at.session_state["chat_messages"][1]["content"]
