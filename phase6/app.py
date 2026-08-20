"""Streamlit chatbot-ui for VoyagerAI Istanbul (Checkpoint Phase 4 D.2B).

Talks ONLY to `agent-system-a`'s public HTTP/SSE API
(`phase6.api_client.SystemAClient`) -- never calls providers, Travel MCP,
Qdrant, or System B directly (architecture.md §4's fixed five-service
topology; the frontend is a leaf that only ever reaches agent-system-a).

The system never books, reserves, or pays for anything -- every rendered
result stays visibly labeled as decision-support information, never a
confirmed reservation.
"""

from __future__ import annotations

import time
import uuid
from datetime import date, datetime, timedelta, timezone
from typing import Any, Optional
from zoneinfo import ZoneInfo

import streamlit as st

# Manual QA remediation Q.1: the same project timezone convention as
# phase4/guards.py::PROJECT_TIMEZONE (services/planner-a) -- "today" for
# date-picker bounds is resolved here too, never the frontend container's
# own local time, so the UI and the backend can never disagree about what
# "today" or "in the past" means for the same wall-clock instant.
_PROJECT_TIMEZONE = ZoneInfo("Europe/Istanbul")

# Manual QA remediation Q.1 (user correction pass §B): TRY and USD --
# genuine FX conversion exists now (providers/fx_frankfurter.py, a real
# Frankfurter.app/ECB rate source, Decimal-exact, with full provenance;
# orchestration/system_a/budget_summary.py applies it server-side). EUR
# remains deferred: no EUR rate source was verified/wired. Keep in sync
# with services/planner-a/phase4/guards.py::_SUPPORTED_CURRENCIES.
_SUPPORTED_CURRENCIES = ["TRY", "USD"]


def _project_today() -> date:
    return datetime.now(timezone.utc).astimezone(_PROJECT_TIMEZONE).date()

from phase6 import results as result_helpers
from phase6.api_client import (
    SSEEvent,
    SystemAClient,
    SystemAConfigurationError,
    SystemAConnectionError,
    SystemAResponseError,
)

st.set_page_config(page_title="VoyagerAI Istanbul", page_icon="🧭", layout="wide")

TERMINAL_STAGES = frozenset({"run_completed", "run_degraded", "run_failed", "run_cancelled"})
STAGE_LABELS = {
    "run_started": "Run started",
    "action_started": "Action started",
    "action_completed": "Action completed",
    "action_failed": "Action failed",
    "run_completed": "Run completed",
    "run_degraded": "Run degraded",
    "run_failed": "Run failed",
    "run_cancelled": "Run cancelled",
}
STATUS_LABELS = {
    "completed": ("✅", "Completed"),
    "degraded": ("⚠️", "Degraded"),
    "failed": ("❌", "Failed"),
    "cancelled": ("🚫", "Cancelled"),
}
# Each rerun performs one bounded read of the event stream (never one
# unbounded blocking read for the whole run) so the "Cancel this run"
# button stays clickable between reruns -- Streamlit has no native
# background-task/websocket primitive this app relies on instead.
_POLL_READ_DEADLINE_SECONDS = 2.0
_POLL_RERUN_DELAY_SECONDS = 0.4


# --- client + session-state plumbing -----------------------------------------------------


@st.cache_resource
def _get_client() -> SystemAClient:
    return SystemAClient()


def _init_session_state() -> None:
    defaults = {
        "run_id": None, "session_id": None, "last_event_id": None,
        "event_log": [], "trip_request_submitted": None,
        # Generated once per "attempt to submit a new trip" and reused
        # for every create_run call until a run is actually created --
        # defense-in-depth against a double-click/double-POST race on
        # top of the structural protection `st.form` + session-state-
        # gated rendering already provide (the form itself disappears
        # from the page the instant a run_id exists, so it cannot be
        # re-submitted by a later, unrelated rerun).
        "pending_idempotency_key": str(uuid.uuid4()),
    }
    for key, value in defaults.items():
        if key not in st.session_state:
            st.session_state[key] = value


def _reset_for_new_trip() -> None:
    """Retrying always creates a brand-new run -- this only clears local
    session state, it never mutates or deletes the completed run's own
    server-side history."""
    for key in ("run_id", "session_id", "last_event_id", "event_log", "trip_request_submitted"):
        st.session_state[key] = None if key != "event_log" else []
    st.session_state["pending_idempotency_key"] = str(uuid.uuid4())


# --- top-level health / mode banner -------------------------------------------------------


def _render_health_banner(client: SystemAClient) -> bool:
    try:
        health = client.health()
    except SystemAConfigurationError as exc:
        st.error(f"System A is not configured: {exc}")
        return False
    except SystemAConnectionError:
        st.error("Cannot reach agent-system-a right now. Please try again shortly.")
        return False
    except SystemAResponseError as exc:
        st.error(f"agent-system-a responded unexpectedly: {exc}")
        return False

    if health.mode == "fixture":
        st.info(
            "🧪 **Demo / fixture mode** — this deployment uses deterministic, network-free demo data. "
            "No live provider, MCP, or A2A call is made, and results shown here are fixture data, "
            "never live availability.",
            icon="🧪",
        )
    return True


def _render_disclaimers() -> None:
    st.caption(
        "VoyagerAI Istanbul is a decision-support tool only — it never books, reserves, or pays for anything. "
        "Accommodation availability shown here is a historical snapshot, not live availability. "
        "Flight and other provider results are point-in-time snapshots, not real-time offers. "
        "A degraded result may omit capabilities that were unavailable when this run executed."
    )


# --- trip request form -------------------------------------------------------------------


_DEPART_DATE_KEY = "voyager_depart_date"
_RETURN_DATE_KEY = "voyager_return_date"


def _clamp_return_date_to_depart() -> None:
    """`on_change` callback for the departure-date widget (Manual QA
    remediation Q.1): runs before the script reruns and re-renders the
    return-date widget below, so if the newly-chosen departure date would
    put the already-stored return date before it, the stored return date
    is pulled forward first -- this is what makes "changing departure
    revalidates return" a live, immediate effect rather than something
    only caught at submit time, and avoids Streamlit raising when a
    widget's `min_value` moves past its own already-stored session value."""
    depart = st.session_state.get(_DEPART_DATE_KEY)
    current_return = st.session_state.get(_RETURN_DATE_KEY)
    if depart is not None and current_return is not None and current_return < depart:
        st.session_state[_RETURN_DATE_KEY] = depart


def _render_trip_form() -> Optional[dict[str, Any]]:
    st.subheader("Plan your Istanbul trip")

    today = _project_today()

    # Deliberately OUTSIDE st.form: a plain widget reruns the script
    # immediately on change (a form's own inputs only take effect on
    # submit), which is what lets the return-date widget's `min_value`
    # below track the live departure-date selection.
    depart_date = st.date_input(
        "Departure date",
        value=max(today + timedelta(days=30), today),
        min_value=today,
        key=_DEPART_DATE_KEY,
        on_change=_clamp_return_date_to_depart,
        help="Cannot be earlier than today (Istanbul time).",
    )

    with st.form("trip_form", clear_on_submit=False):
        col1, col2 = st.columns(2)
        with col1:
            origin = st.text_input(
                "Origin airport (IATA code)", value="BEY", max_chars=3,
                help="3-letter IATA code, e.g. BEY for Beirut.",
            ).strip().upper()
            traveler_count = st.number_input("Travelers", min_value=1, max_value=12, value=2, step=1)
            pace = st.selectbox("Pace", options=["relaxed", "moderate", "packed"], index=1)
        with col2:
            st.text_input("Destination", value="Istanbul (IST)", disabled=True, help="Istanbul is the only fully supported destination.")
            return_date = st.date_input(
                "Return date",
                value=max(st.session_state.get(_RETURN_DATE_KEY, today + timedelta(days=35)), depart_date),
                min_value=depart_date,
                key=_RETURN_DATE_KEY,
                help="Cannot be earlier than the departure date.",
            )
            currency = st.selectbox(
                "Currency", options=_SUPPORTED_CURRENCIES, index=0,
                help="TRY prices are native. USD amounts are converted using one real, live exchange rate per run — see the Budget and Sources tabs for the rate and its date.",
            )
            # Pre-commit stabilization: the label was previously
            # f"Budget ({currency})" -- inside this st.form, changing the
            # Currency selectbox above does not trigger an immediate
            # rerun (form widgets only rerun on submit), so that label
            # kept showing the PREVIOUS currency while the user was still
            # interacting with the form, even though the value actually
            # submitted was always correctly built from the live
            # `currency` variable at submit time. A stable label sidesteps
            # the misleading stale-currency display entirely; the caption
            # states the rule once, plainly, instead of implying the label
            # itself tracks the live selection.
            budget_amount = st.number_input(
                "Budget amount", min_value=1.0, value=5000.0, step=100.0,
                help="Enter the amount in whichever currency you selected above.",
            )
            st.caption("Uses the currency selected above.")
            language = st.selectbox("Preferred language", options=["en", "tr", "ar"], index=0, help="Used for local-expertise citations where the underlying knowledge base supports it.")

        interests = st.multiselect(
            "Interests", options=["history", "food", "shopping", "nightlife", "art", "nature", "family"],
            default=["history"],
        )
        mobility_constraints = st.multiselect(
            "Accessibility / mobility preferences",
            options=["wheelchair_accessible", "limited_walking", "step_free_only"],
            default=[],
        )
        user_message = st.text_area(
            "Tell us more about your trip (free text)", value="Plan a comfortable trip to Istanbul.", max_chars=4000,
        )

        submitted = st.form_submit_button("Plan my trip", width="stretch")

    if not submitted:
        return None

    if len(origin) != 3 or not origin.isalpha():
        st.error("Origin airport must be a 3-letter IATA code.")
        return None
    # Defense in depth alongside the widgets' own min_value constraints
    # above (Manual QA remediation Q.1): a page left open across a day
    # boundary, or a widget value restored from stale session state,
    # could still submit an out-of-range date even though the picker
    # itself steers a live interaction away from one. The System A API
    # re-validates these exact same rules server-side and rejects with a
    # typed error before creating a run either way -- this is a fast,
    # friendly local echo of that same validation, not the only guard.
    if depart_date < today:
        st.error("Departure date cannot be in the past.")
        return None
    if return_date < depart_date:
        st.error("Return date cannot be before the departure date.")
        return None

    trip_request = {
        "origin": origin, "destination": "IST",
        "depart_date": depart_date.isoformat(), "return_date": return_date.isoformat(),
        "traveler_count": int(traveler_count),
        "budget": {"amount_minor_units": int(round(budget_amount * 100)), "currency": currency},
        "preferences": {
            "interests": interests, "pace": pace, "language": language,
            "mobility_constraints": mobility_constraints,
        },
    }
    return {"user_message": user_message.strip() or "Plan a trip to Istanbul.", "trip_request": trip_request}


# --- progress streaming ------------------------------------------------------------------


def _pull_new_events(
    client: SystemAClient, run_id: str, last_event_id: Optional[str], deadline_seconds: float = _POLL_READ_DEADLINE_SECONDS
) -> tuple[list[SSEEvent], Optional[str]]:
    """One bounded read of the SSE stream: collects events for at most
    `deadline_seconds` (or until a terminal event arrives), then returns
    -- never blocks for the whole run duration in one call, so the page
    stays interactive between reruns."""
    collected: list[SSEEvent] = []
    started = time.monotonic()
    try:
        for event in client.stream_events(run_id, last_event_id=last_event_id):
            collected.append(event)
            last_event_id = event.event_id
            if event.stage in TERMINAL_STAGES or (time.monotonic() - started) >= deadline_seconds:
                break
    except (SystemAConnectionError, SystemAResponseError):
        pass  # transient -- the next rerun retries
    return collected, last_event_id


def _render_progress_log() -> None:
    if not st.session_state["event_log"]:
        st.write("Waiting for progress…")
        return
    last_stage = st.session_state["event_log"][-1]["stage"]
    is_terminal = last_stage in TERMINAL_STAGES
    with st.status("Planning your trip…", expanded=not is_terminal) as status:
        for entry in st.session_state["event_log"]:
            label = STAGE_LABELS.get(entry["stage"], entry["stage"])
            action = (entry.get("payload") or {}).get("action")
            st.write(label + (f" — `{action}`" if action else ""))
        if is_terminal:
            key = last_stage.replace("run_", "")
            icon, text = STATUS_LABELS.get(key, ("ℹ️", last_stage))
            status.update(label=f"{icon} {text}", state=("error" if key == "failed" else "complete"))
        else:
            status.update(label="Planning your trip…", state="running")


# --- terminal-result rendering ------------------------------------------------------------


def _render_money(label: str, money: Optional[dict], result: dict, *, bold: bool = True) -> None:
    """The one shared rendering call for every price in the app (Manual
    QA remediation Q.1, second correction pass, §1) -- always shows the
    trip's own currency as the primary, headline number when
    determinable, discloses the real raw provider-native amount as a
    caption when it differs, and shows an explicit warning (never a
    silently-mislabeled amount) when conversion was needed but
    unavailable."""
    pair = result_helpers.format_money_pair(money, result)
    primary = f"**{pair['primary']}**" if bold else pair["primary"]
    st.write(f"{label}: {primary}")
    if pair["secondary"]:
        st.caption(pair["secondary"])
    if pair["conversion_unavailable"]:
        target = result_helpers.get_target_currency(result)
        st.caption(f"⚠️ Live {target} conversion unavailable for this amount — showing the real, original currency only.")


def _render_flights(result: dict) -> None:
    options = result_helpers.extract_flight_options(result)
    if not options:
        st.info("No flight results are available for this run.")
        return
    for opt in options:
        with st.container(border=True):
            st.markdown(f"**{opt.get('carrier', 'Unknown carrier')}** — {opt.get('origin', '?')} → {opt.get('destination', '?')}")
            st.write(f"Depart: {opt.get('depart_at', '—')}  ·  Arrive: {opt.get('arrive_at', '—')}  ·  Stops: {opt.get('stops', '—')}")
            _render_money("Price", opt.get("price"), result)


def _render_stays(result: dict, trip_request: Optional[dict]) -> None:
    items = result_helpers.extract_stay_items(result)
    if not items:
        st.info("No accommodation results are available for this run.")
        return
    st.warning("Accommodation availability shown is a historical snapshot — never live availability.", icon="📦")
    nights = result_helpers.nights_from_trip_request(trip_request)
    for item in items:
        stay = item.get("stay") or {}
        fair_price = item.get("fair_price") or {}
        nightly_price = stay.get("nightly_price")
        with st.container(border=True):
            st.markdown(f"**{stay.get('name', 'Unnamed stay')}** ({stay.get('district_id', '—')}, {stay.get('side', '—')} side)")
            _render_money("Nightly price", nightly_price, result)
            st.caption(f"Rank: {item.get('rank', '—')}")

            if isinstance(nightly_price, dict) and nightly_price.get("amount_minor_units") is not None and nights:
                total_raw = {"amount_minor_units": nightly_price["amount_minor_units"] * nights, "currency": nightly_price.get("currency")}
                _render_money(f"Total for {nights} night(s)", total_raw, result, bold=False)

            fair_amount = fair_price.get("estimated_fair_price")
            if isinstance(fair_amount, dict):
                _render_money("Estimated fair price", fair_amount, result, bold=False)
                st.caption(f"Scoring status: {fair_price.get('scoring_status', 'unknown')}")
                _render_savings(nightly_price, fair_amount, result)


def _render_savings(nightly_price: Optional[dict], fair_price: Optional[dict], result: dict) -> None:
    """The deal/savings figure (Manual QA remediation Q.1, second
    correction pass, §1) -- nightly price vs. estimated fair price,
    computed only after both are normalized into the SAME currency (the
    run's one FX quote, never a second call), so the comparison is never
    between two different currencies. Silent (never a fabricated number)
    if either side can't be normalized into a common currency."""
    target = result_helpers.get_target_currency(result)
    fx_quote = result_helpers.get_fx_quote(result)
    if target is None:
        # No trip-level currency context (e.g. no trip_request at all) --
        # only compare directly when both are already the same currency.
        if not (isinstance(nightly_price, dict) and isinstance(fair_price, dict)):
            return
        if str(nightly_price.get("currency")).upper() != str(fair_price.get("currency")).upper():
            return
        normalized_nightly, normalized_fair = nightly_price, fair_price
    else:
        normalized_nightly = result_helpers.normalize_money(nightly_price, target, fx_quote)
        normalized_fair = result_helpers.normalize_money(fair_price, target, fx_quote)
    if normalized_nightly is None or normalized_fair is None:
        return
    diff_minor_units = normalized_nightly["amount_minor_units"] - normalized_fair["amount_minor_units"]
    diff_money = {"amount_minor_units": abs(diff_minor_units), "currency": normalized_nightly["currency"]}
    if diff_minor_units < 0:
        st.caption(f"💚 {result_helpers.format_money(diff_money)} below the estimated fair price.")
    elif diff_minor_units > 0:
        st.caption(f"🔺 {result_helpers.format_money(diff_money)} above the estimated fair price.")


# Manual QA remediation Q.1: precise, honest per-status explanations --
# never the bare, vague status code itself. "forecast_not_yet_available"
# is the one that most needs a specific explanation (the manual-run
# observation this fixes): a trip date beyond Open-Meteo's forecast
# horizon is expected behavior, not a provider outage, and the UI should
# say so with the exact date forecasts open up.
_WEATHER_STATUS_MESSAGES = {
    "unsupported": "Weather is not available for this request (an unsupported date range or location).",
    "timeout": "Weather could not be retrieved right now — the forecast provider timed out. This is not a permanent failure; try again shortly.",
    "rate_limited": "Weather could not be retrieved right now — the forecast provider is temporarily rate-limited. Try again shortly.",
    "provider_error": "Weather could not be retrieved right now — the forecast provider returned an unexpected error.",
    "unavailable": "Weather could not be retrieved right now — the forecast provider was temporarily unavailable.",
    "cancelled": "The weather lookup was cancelled before it completed.",
}


def _render_forecast_days(days: list[dict]) -> None:
    for day in days or []:
        st.write(f"**{day.get('date', '—')}** — {day.get('condition', '—')}, high {day.get('high', '—')}° / low {day.get('low', '—')}°")


def _render_historical_climate_days(weather: dict) -> None:
    # User correction pass (§C): a future trip beyond the live forecast
    # horizon gets real historical climate guidance -- always clearly
    # labeled as an aggregate, never presented as if it were a forecast.
    years = weather.get("years_sampled") or []
    st.warning(weather.get("climate_disclaimer") or "Historical climate guidance — not a forecast.")
    if years:
        st.caption(f"Aggregated from {len(years)} prior year(s): {', '.join(str(y) for y in years)}.")
    earliest = weather.get("earliest_available_forecast_date")
    if earliest:
        st.caption(f"A live forecast for this trip date opens up on **{earliest}**.")
    for day in weather.get("historical_climate_days") or []:
        st.write(
            f"**{day.get('date', '—')}** (historical average) — {day.get('condition', '—')}, "
            f"avg high {day.get('avg_high', '—')}° / avg low {day.get('avg_low', '—')}°"
        )


def _render_weather(result: dict) -> None:
    weather = result_helpers.extract_weather(result)
    if weather:
        kind = weather.get("kind")
        if kind == "forecast":
            _render_forecast_days(weather.get("forecast_days") or [])
        elif kind == "current_observation":
            obs = weather.get("observation") or {}
            st.write(f"Current conditions: {obs.get('condition', '—')}")
        elif kind == "historical":
            _render_historical_climate_days(weather)
        elif kind == "mixed":
            st.caption("Coverage: live forecast for the reachable dates, historical climate guidance for the rest.")
            st.write("**Live forecast**")
            _render_forecast_days(weather.get("forecast_days") or [])
            st.write("**Historical climate guidance**")
            _render_historical_climate_days(weather)
        else:
            st.json(weather)
        return

    status_info = result_helpers.extract_weather_status(result)
    if status_info is None:
        st.info("No weather results are available for this run.")
        return

    status = status_info.get("status")
    st.info(_WEATHER_STATUS_MESSAGES.get(status, "No weather results are available for this run."))


def _render_itinerary(result: dict) -> None:
    itinerary = result_helpers.extract_itinerary(result)
    if not itinerary:
        st.info("No day-by-day itinerary is available for this run.")
        return
    st.write(f"Recommended base: `{itinerary.get('recommended_base_candidate_id', '—')}`  ·  Side crossings: {itinerary.get('side_crossings', '—')}")
    for day in itinerary.get("daily_plans") or []:
        with st.expander(f"📅 {day.get('date', '—')} — {day.get('side', '—')} side", expanded=False):
            poi_ids = day.get("poi_ids") or []
            st.write("Points of interest: " + (", ".join(f"`{p}`" for p in poi_ids) if poi_ids else "—"))
            st.write(
                f"Walking: {day.get('walking_minutes', '—')} min · Transfers: {day.get('transfer_minutes', '—')} min "
                f"· Activities: {day.get('activity_minutes', '—')} min · Slack: {day.get('slack_minutes', '—')} min"
            )
            for warning in day.get("warnings") or []:
                st.warning(warning)


def _render_fx_disclosure(result: dict) -> None:
    # Manual QA remediation Q.1 (§B): "the Budget and Sources tabs must
    # disclose the FX source and rate date" -- always renders something
    # explicit, including an honest degradation message when the FX
    # provider failed rather than staying silent about it.
    budget_summary = result.get("budget_summary") if isinstance(result, dict) else None
    if not isinstance(budget_summary, dict):
        return
    quote = budget_summary.get("fx_quote")
    if isinstance(quote, dict):
        st.caption(
            f"Exchange rate: 1 {quote['base_currency']} = {quote['rate']} {quote['quote_currency']} "
            f"(as of {quote['effective_date']}, source: {quote['provider']}, retrieved {quote['retrieved_at']}, "
            f"cache: {quote['cache_status']})."
        )
    elif budget_summary.get("fx_status") is not None:
        st.warning(
            f"A live exchange rate could not be obtained for this run (status: {budget_summary['fx_status']}) -- "
            "amounts are shown in their real, original currency only, never a fabricated conversion."
        )


def _render_budget_chart(result: dict, trip_request: Optional[dict]) -> None:
    chart_data = result_helpers.build_budget_chart_data(result, trip_request)
    if chart_data is None:
        st.info("Not enough real numeric data was returned to build a cost overview for this run.")
        _render_fx_disclosure(result)
        return
    try:
        import plotly.graph_objects as go
    except ImportError:
        st.info("Plotly is not available in this environment.")
        return
    figure = go.Figure(data=[go.Bar(x=chart_data["labels"], y=chart_data["values"])])
    figure.update_layout(
        title=f"Cost overview ({chart_data['currency']})", yaxis_title=chart_data["currency"] or "Amount",
        height=360, margin=dict(l=10, r=10, t=40, b=10),
    )
    st.plotly_chart(figure, width="stretch")
    st.caption("Built only from numeric values actually returned by this run — never estimated or invented.")
    _render_fx_disclosure(result)


def _render_map(result: dict) -> None:
    points = result_helpers.extract_map_points(result)
    if not points:
        st.info("No trusted coordinates were returned for this run, so no map is shown.")
        return
    try:
        import folium
        from streamlit_folium import st_folium
    except ImportError:
        st.info("Map rendering is not available in this environment.")
        return
    avg_lat = sum(p["lat"] for p in points) / len(points)
    avg_lon = sum(p["lon"] for p in points) / len(points)
    fmap = folium.Map(location=[avg_lat, avg_lon], zoom_start=13)
    for point in points:
        # Manual QA remediation Q.1 (second correction pass, §1): the
        # popup's primary price is normalized into the trip's own
        # currency (never a second FX call -- format_money_pair reuses
        # the run's one already-fetched quote); the raw amount is kept
        # visible in the popup text as secondary detail, never hidden.
        pair = result_helpers.format_money_pair(point.get("nightly_price_raw"), result)
        price_text = pair["primary"]
        if pair["secondary"]:
            price_text += f" ({pair['secondary']})"
        elif pair["conversion_unavailable"]:
            price_text += " (live conversion unavailable)"
        folium.Marker(
            location=[point["lat"], point["lon"]],
            popup=f"{point['label']} — {price_text}",
            tooltip=point["label"],
        ).add_to(fmap)
    st_folium(fmap, use_container_width=True, height=420, returned_objects=[])


def _render_provenance_and_citations(result: dict) -> None:
    badges = result_helpers.collect_provenance_badges(result)
    if badges:
        st.markdown("**Data sources per action**")
        for badge in badges:
            mode = badge.get("data_mode") or "unknown"
            st.write(f"- `{badge['action']}` — status: **{badge['status']}**, data mode: **{mode}**" + (f", provider: {badge['provider']}" if badge.get("provider") else ""))
    citations = result_helpers.collect_citations(result)
    if citations:
        st.markdown("**Citations**")
        for citation in citations:
            title = citation.get("title") or citation.get("source_id", "Source")
            st.write(f"- {title}")

    assumptions = result_helpers.collect_itinerary_assumptions(result)
    if assumptions:
        st.markdown("**Assumptions & limitations**")
        for assumption in assumptions:
            st.write(f"- {assumption}")
    elif not citations and result_helpers.extract_itinerary(result):
        # An itinerary was returned with neither citations nor a stated
        # assumption -- an honest, explicit gap, never silently blank.
        st.caption("No citations were returned for this itinerary, and no explicit reason was given.")

    if not badges and not citations and not assumptions:
        st.info("No provenance or citation information is available for this run.")

    _render_fx_disclosure(result)


def _render_results(result: dict, trip_request: Optional[dict]) -> None:
    warnings = result_helpers.collect_warnings(result)
    if warnings:
        with st.expander(f"⚠️ Warnings ({len(warnings)})", expanded=True):
            for warning in warnings:
                st.write(f"- {warning}")

    if result.get("narrative"):
        st.markdown(f"**Summary:** {result['narrative']}")

    tabs = st.tabs(["✈️ Flights", "🏨 Stays & fair price", "🌤️ Weather", "🗺️ Itinerary", "💰 Budget", "📍 Map", "📚 Sources"])
    with tabs[0]:
        _render_flights(result)
    with tabs[1]:
        _render_stays(result, trip_request)
    with tabs[2]:
        _render_weather(result)
    with tabs[3]:
        _render_itinerary(result)
    with tabs[4]:
        _render_budget_chart(result, trip_request)
    with tabs[5]:
        _render_map(result)
    with tabs[6]:
        _render_provenance_and_citations(result)


def _render_terminal_state(status: str, result: dict) -> None:
    icon, label = STATUS_LABELS.get(status, ("ℹ️", status))
    if status == "completed":
        st.success(f"{icon} {label}")
    elif status == "degraded":
        st.warning(f"{icon} {label} — some information could not be retrieved; a partial plan is shown below.")
    elif status == "cancelled":
        st.warning(f"{icon} {label} — this run was cancelled before completing.")
    else:
        st.error(f"{icon} {label} — {result.get('reason', 'an internal error occurred')}")
        return

    _render_disclaimers()
    _render_results(result, st.session_state.get("trip_request_submitted"))


# --- run section (progress + terminal) -----------------------------------------------------


def _render_run_section(client: SystemAClient) -> None:
    run_id = st.session_state["run_id"]
    header_col, action_col = st.columns([3, 1])
    with header_col:
        st.info(f"**Run ID:** `{run_id}`  ·  **Session ID:** `{st.session_state['session_id']}`")
    with action_col:
        if st.button("Plan a new trip", width="stretch", key="new_trip_btn"):
            _reset_for_new_trip()
            st.rerun()

    try:
        current = client.get_run(run_id)
    except SystemAConnectionError:
        st.error("Cannot reach agent-system-a to check run status. It may come back online shortly.")
        return
    except SystemAResponseError as exc:
        st.error(f"Could not retrieve run status: {exc}")
        return

    if current.status not in ("pending", "running"):
        _render_progress_log()
        _render_terminal_state(current.status, current.result or {})
        return

    with header_col:
        if st.button("Cancel this run", key="cancel_btn"):
            try:
                client.cancel_run(run_id)
                st.toast("Cancellation requested.")
            except (SystemAConnectionError, SystemAResponseError) as exc:
                st.warning(f"Could not request cancellation: {exc}")
            st.rerun()

    _render_progress_log()
    new_events, new_last_id = _pull_new_events(client, run_id, st.session_state["last_event_id"])
    if new_events:
        st.session_state["event_log"].extend({"stage": e.stage, "payload": e.payload} for e in new_events)
        st.session_state["last_event_id"] = new_last_id
    time.sleep(_POLL_RERUN_DELAY_SECONDS)
    st.rerun()


# --- entrypoint ---------------------------------------------------------------------------


def main() -> None:
    _init_session_state()
    st.title("🧭 VoyagerAI Istanbul")
    st.caption("A decision-support planner for Istanbul trips. This system never books, reserves, or pays for anything.")

    client = _get_client()
    if not _render_health_banner(client):
        return

    if st.session_state["run_id"] is None:
        form_result = _render_trip_form()
        if form_result is None:
            return
        try:
            created = client.create_run(
                user_message=form_result["user_message"],
                trip_request=form_result["trip_request"],
                idempotency_key=st.session_state["pending_idempotency_key"],
            )
        except SystemAConnectionError:
            st.error("Could not reach agent-system-a. Please try again.")
            return
        except SystemAResponseError as exc:
            st.error(f"Could not start a planning run: {exc}")
            return
        st.session_state["run_id"] = created.run_id
        st.session_state["session_id"] = created.session_id
        st.session_state["trip_request_submitted"] = form_result["trip_request"]
        st.rerun()
        return

    _render_run_section(client)


main()
