"""Pure, Streamlit-free extraction/formatting helpers over a validated
System A `final_result` dict (Checkpoint Phase 4 D.2B). Every function
here is defensive: a missing optional section returns an empty
list/None, never raises, never fabricates a value that was not actually
present in the API's own response. Kept separate from `app.py` so this
logic is unit-testable without Streamlit's AppTest.

`final_result` is exactly `phase4.graph`'s own
`{"status", "observations", "warnings", "narrative"}` (or
`{"status", "reason", "observations", "warnings"}` for a degraded/
cancelled result) shape, reused unmodified -- this module never redefines
what a result *is*, only reads it.
"""

from __future__ import annotations

from typing import Any, Optional

# search_stays/estimate_fair_price/call_istanbul_expert observations are
# NOT ProviderResponseEnvelope-wrapped (ADR 0009 §5 / ADR 0014 §3) --
# their `envelope` field IS the raw result shape directly. Every other
# action's `envelope` is a real ProviderResponseEnvelope whose own
# `result` field holds the capability-specific payload.
_UNWRAPPED_ACTIONS = frozenset({"search_stays", "estimate_fair_price", "call_istanbul_expert"})


def observations_by_action(result: Optional[dict[str, Any]]) -> dict[str, list[dict[str, Any]]]:
    if not result or not isinstance(result, dict):
        return {}
    grouped: dict[str, list[dict[str, Any]]] = {}
    for obs in result.get("observations") or []:
        if isinstance(obs, dict) and obs.get("action"):
            grouped.setdefault(obs["action"], []).append(obs)
    return grouped


def _successful_capability_payload(observation: dict[str, Any]) -> Optional[dict[str, Any]]:
    if observation.get("status") != "success":
        return None
    envelope = observation.get("envelope")
    if not isinstance(envelope, dict):
        return None
    if observation.get("action") in _UNWRAPPED_ACTIONS:
        return envelope
    return envelope.get("result") if isinstance(envelope.get("result"), dict) else None


def extract_flight_options(result: Optional[dict[str, Any]]) -> list[dict[str, Any]]:
    grouped = observations_by_action(result)
    options: list[dict[str, Any]] = []
    for obs in grouped.get("search_flights", []):
        payload = _successful_capability_payload(obs)
        if payload:
            options.extend(o for o in payload.get("options") or [] if isinstance(o, dict))
    return options


def extract_stay_items(result: Optional[dict[str, Any]]) -> list[dict[str, Any]]:
    grouped = observations_by_action(result)
    items: list[dict[str, Any]] = []
    for obs in grouped.get("search_stays", []):
        payload = _successful_capability_payload(obs)
        if payload:
            items.extend(i for i in payload.get("stays") or [] if isinstance(i, dict))
    return items


def extract_weather(result: Optional[dict[str, Any]]) -> Optional[dict[str, Any]]:
    grouped = observations_by_action(result)
    for obs in grouped.get("get_weather", []):
        payload = _successful_capability_payload(obs)
        if payload:
            return payload
    return None


def extract_web_evidence(result: Optional[dict[str, Any]]) -> Optional[dict[str, Any]]:
    grouped = observations_by_action(result)
    for obs in grouped.get("web_search", []):
        payload = _successful_capability_payload(obs)
        if payload:
            return payload
    return None


def extract_itinerary(result: Optional[dict[str, Any]]) -> Optional[dict[str, Any]]:
    grouped = observations_by_action(result)
    for obs in grouped.get("call_istanbul_expert", []):
        payload = _successful_capability_payload(obs)
        if payload:
            return payload
    return None


def extract_fair_price_items(result: Optional[dict[str, Any]]) -> list[dict[str, Any]]:
    """Fair-price info arrives two ways: nested inside each `search_stays`
    item (`item["fair_price"]`) and, if the caller separately asked for
    it, a standalone `estimate_fair_price` observation. Both are
    collected, never invented when absent."""
    items: list[dict[str, Any]] = []
    for stay_item in extract_stay_items(result):
        fair_price = stay_item.get("fair_price")
        if isinstance(fair_price, dict):
            items.append(fair_price)
    grouped = observations_by_action(result)
    for obs in grouped.get("estimate_fair_price", []):
        payload = _successful_capability_payload(obs)
        if payload and isinstance(payload.get("fair_price"), dict):
            items.append(payload["fair_price"])
    return items


def collect_warnings(result: Optional[dict[str, Any]]) -> list[str]:
    if not result:
        return []
    warnings: list[str] = list(result.get("warnings") or [])
    for observations in observations_by_action(result).values():
        for obs in observations:
            warnings.extend(obs.get("warnings") or [])
    itinerary = extract_itinerary(result)
    if itinerary:
        warnings.extend(itinerary.get("warnings") or [])
    seen: set[str] = set()
    deduped = []
    for w in warnings:
        if w not in seen:
            seen.add(w)
            deduped.append(w)
    return deduped


def collect_citations(result: Optional[dict[str, Any]]) -> list[dict[str, Any]]:
    itinerary = extract_itinerary(result)
    if not itinerary:
        return []
    return [c for c in itinerary.get("citations") or [] if isinstance(c, dict)]


def collect_itinerary_assumptions(result: Optional[dict[str, Any]]) -> list[str]:
    """`LocalItinerary.assumptions` -- e.g. the explicit "this is a
    deterministic fixture, not a real A2A call to System B" statement
    fixture mode always carries when `citations` is empty. A real defect
    found during this checkpoint's own browser smoke test: this field
    was already returned by the API but was never rendered anywhere,
    leaving "why are there no citations?" unanswered on screen even
    though the data itself always explains it."""
    itinerary = extract_itinerary(result)
    if not itinerary:
        return []
    return [a for a in itinerary.get("assumptions") or [] if isinstance(a, str)]


def collect_provenance_badges(result: Optional[dict[str, Any]]) -> list[dict[str, Any]]:
    """One badge per observation: action, status, data_mode (if known),
    provider (if known) -- drawn only from fields already present in the
    response, never inferred."""
    badges: list[dict[str, Any]] = []
    if not result:
        return badges
    for obs in result.get("observations") or []:
        if not isinstance(obs, dict):
            continue
        action = obs.get("action", "unknown")
        status = obs.get("status", "unknown")
        data_mode = None
        provider = None
        envelope = obs.get("envelope")
        if isinstance(envelope, dict):
            data_mode = envelope.get("data_mode")
            provider = envelope.get("provider")
            if data_mode is None and isinstance(envelope.get("stay"), (dict, type(None))):
                pass  # search_stays' own top-level shape has no single data_mode -- left None, never guessed
        badges.append({"action": action, "status": status, "data_mode": data_mode, "provider": provider})
    return badges


def format_money(money: Optional[dict[str, Any]]) -> str:
    if not isinstance(money, dict):
        return "—"
    amount = money.get("amount_minor_units")
    currency = money.get("currency")
    if amount is None or not currency:
        return "—"
    try:
        return f"{amount / 100:,.2f} {currency}"
    except (TypeError, ValueError):
        return "—"


def extract_map_points(result: Optional[dict[str, Any]]) -> list[dict[str, Any]]:
    """Only ever reads `coordinates.lat`/`coordinates.lon` fields the API
    already returned on a real stay candidate -- never geocodes, never
    estimates, never invents a coordinate."""
    points: list[dict[str, Any]] = []
    for item in extract_stay_items(result):
        stay = item.get("stay")
        if not isinstance(stay, dict):
            continue
        coords = stay.get("coordinates")
        if not isinstance(coords, dict) or "lat" not in coords or "lon" not in coords:
            continue
        try:
            lat, lon = float(coords["lat"]), float(coords["lon"])
        except (TypeError, ValueError):
            continue
        points.append({
            "label": stay.get("name") or stay.get("stay_id") or "Stay candidate",
            "lat": lat, "lon": lon,
            "nightly_price": format_money(stay.get("nightly_price")),
        })
    return points


def build_budget_chart_data(
    result: Optional[dict[str, Any]], trip_request: Optional[dict[str, Any]]
) -> Optional[dict[str, list]]:
    """A cost overview built ONLY from numeric values actually present:
    the traveler's own submitted budget (echoed back from the request
    they typed, never invented), the cheapest returned flight price, and
    the cheapest returned nightly stay price x number of nights. Returns
    None when there is not enough real numeric data to chart anything
    honest -- never a chart with a fabricated bar."""
    labels: list[str] = []
    values: list[float] = []
    currency = None

    if isinstance(trip_request, dict) and isinstance(trip_request.get("budget"), dict):
        budget = trip_request["budget"]
        if budget.get("amount_minor_units") is not None and budget.get("currency"):
            currency = budget["currency"]
            labels.append("Your budget")
            values.append(budget["amount_minor_units"] / 100)

    flights = extract_flight_options(result)
    flight_prices = [f["price"]["amount_minor_units"] for f in flights if isinstance(f.get("price"), dict) and f["price"].get("amount_minor_units") is not None]
    if flight_prices:
        labels.append("Cheapest flight")
        values.append(min(flight_prices) / 100)

    stays = extract_stay_items(result)
    nights = None
    if isinstance(trip_request, dict) and trip_request.get("depart_date") and trip_request.get("return_date"):
        try:
            from datetime import date as _date
            depart = _date.fromisoformat(str(trip_request["depart_date"]))
            ret = _date.fromisoformat(str(trip_request["return_date"]))
            nights = max((ret - depart).days, 0)
        except ValueError:
            nights = None
    nightly_prices = [
        s["stay"]["nightly_price"]["amount_minor_units"]
        for s in stays
        if isinstance(s.get("stay"), dict) and isinstance(s["stay"].get("nightly_price"), dict)
        and s["stay"]["nightly_price"].get("amount_minor_units") is not None
    ]
    if nightly_prices and nights:
        labels.append("Cheapest stay total")
        values.append(min(nightly_prices) * nights / 100)

    if len(values) < 2:  # need at least a budget + one real cost to be a meaningful comparison
        return None
    return {"labels": labels, "values": values, "currency": currency or ""}
