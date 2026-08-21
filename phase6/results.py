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

from decimal import Decimal, InvalidOperation, ROUND_HALF_UP
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


def extract_weather_status(result: Optional[dict[str, Any]]) -> Optional[dict[str, Any]]:
    """Manual QA remediation Q.1: unlike `extract_weather` (successful
    results only), this returns the most recent get_weather observation's
    raw status and envelope result even when NOT successful -- lets the UI
    explain precisely why weather is unavailable (e.g.
    'forecast_not_yet_available' with the exact date it opens up) instead
    of staying silent or showing a bare, unexplained status code."""
    grouped = observations_by_action(result)
    weather_obs = grouped.get("get_weather") or []
    if not weather_obs:
        return None
    latest = weather_obs[-1]
    envelope = latest.get("envelope")
    envelope_result = envelope.get("result") if isinstance(envelope, dict) else None
    return {"status": latest.get("status"), "result": envelope_result}


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


def candidate_provenance_by_poi_id(result: Optional[dict[str, Any]]) -> dict[str, dict[str, Any]]:
    """`LocalItinerary.candidate_provenance` (RAG-FIRST SYSTEM B R.1;
    additive, optional) indexed by poi_id -- empty dict when the field is
    absent entirely (an itinerary payload produced before this
    checkpoint), never fabricated."""
    itinerary = extract_itinerary(result)
    if not itinerary:
        return {}
    entries = itinerary.get("candidate_provenance")
    if not isinstance(entries, list):
        return {}
    return {e["poi_id"]: e for e in entries if isinstance(e, dict) and isinstance(e.get("poi_id"), str)}


def humanize_poi_id(poi_id: str) -> str:
    """Deterministic last-resort label when no real display_name is
    available (a catalog_fallback candidate never carries one, and an
    older payload may carry no candidate_provenance at all) -- turns
    'poi_hagia_sophia' into 'Hagia Sophia'. Never used when a real
    display_name is present; this is presentation only, never a
    substitute for real provenance data."""
    if not isinstance(poi_id, str) or not poi_id:
        return str(poi_id)
    stripped = poi_id[len("poi_") :] if poi_id.startswith("poi_") else poi_id
    words = [w for w in stripped.split("_") if w]
    return " ".join(w.capitalize() for w in words) or poi_id


def poi_display_name(poi_id: str, provenance_by_id: dict[str, dict[str, Any]]) -> str:
    """The name to show a user for this POI -- never the raw poi_*
    identifier as the primary label. Prefers the real display_name
    carried on a 'rag'-origin candidate_provenance entry; falls back to
    a humanized form of the id itself (never to the raw id string)."""
    entry = provenance_by_id.get(poi_id)
    if isinstance(entry, dict):
        name = entry.get("display_name")
        if isinstance(name, str) and name.strip():
            return name
    return humanize_poi_id(poi_id)


def poi_origin_badge(poi_id: str, provenance_by_id: dict[str, dict[str, Any]]) -> Optional[str]:
    """'RAG' or 'Catalog' -- or None when no provenance is known at all
    (an itinerary payload from before candidate_provenance existed).
    Never guesses an origin the API did not actually report."""
    entry = provenance_by_id.get(poi_id)
    if not isinstance(entry, dict):
        return None
    origin = entry.get("candidate_origin")
    if origin == "rag":
        return "RAG"
    if origin == "catalog_fallback":
        return "Catalog"
    return None


def poi_matched_interests_text(poi_id: str, provenance_by_id: dict[str, dict[str, Any]]) -> Optional[str]:
    """Comma-joined matched_interests for this POI, or None when absent
    (a catalog_fallback entry never carries this field, and a
    'rag'-origin entry from before this checkpoint may not either)."""
    entry = provenance_by_id.get(poi_id)
    if not isinstance(entry, dict):
        return None
    matched = entry.get("matched_interests")
    if not isinstance(matched, list) or not matched:
        return None
    return ", ".join(str(m) for m in matched if isinstance(m, str))


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


def get_target_currency(result: Optional[dict[str, Any]]) -> Optional[str]:
    """The trip's own budget currency, if a real budget_summary exists
    for this run (orchestration/system_a/budget_summary.py) -- Manual QA
    remediation Q.1 (second correction pass, §1)."""
    budget_summary = result.get("budget_summary") if isinstance(result, dict) else None
    if not isinstance(budget_summary, dict):
        return None
    budget = budget_summary.get("budget")
    currency = budget.get("currency") if isinstance(budget, dict) else None
    return str(currency).upper() if currency else None


def get_fx_quote(result: Optional[dict[str, Any]]) -> Optional[dict[str, Any]]:
    """The ONE real, already-fetched run-level FX quote for this run, or
    None if no conversion was ever needed/possible -- every UI location
    that normalizes an amount reuses THIS quote, never issuing a second
    FX call of its own (Manual QA remediation Q.1, second correction
    pass, §1)."""
    budget_summary = result.get("budget_summary") if isinstance(result, dict) else None
    if not isinstance(budget_summary, dict):
        return None
    quote = budget_summary.get("fx_quote")
    return quote if isinstance(quote, dict) else None


def normalize_money(
    money: Optional[dict[str, Any]], target_currency: Optional[str], fx_quote: Optional[dict[str, Any]],
) -> Optional[dict[str, Any]]:
    """Converts a real, raw Money-shaped dict into `target_currency`
    using the already-fetched `fx_quote` -- pure arithmetic on data the
    API already returned, never a network call of its own. Returns the
    input unchanged (as a new dict; `money` itself is never mutated) when
    its currency already matches `target_currency`. Returns None -- never
    a fabricated amount -- when conversion isn't possible: no quote, or a
    currency pair the quote doesn't actually cover. `Decimal`/
    `ROUND_HALF_UP` throughout, mirroring providers/money.py's own
    conversion rule exactly (this module cannot import from `providers/`
    -- the frontend never calls a provider directly -- so the same small,
    pure arithmetic is reproduced here rather than reused)."""
    if not isinstance(money, dict):
        return None
    amount = money.get("amount_minor_units")
    currency = money.get("currency")
    if amount is None or not currency or not target_currency:
        return None
    currency = str(currency).upper()
    target = str(target_currency).upper()
    if currency == target:
        return {"amount_minor_units": amount, "currency": target}
    if not isinstance(fx_quote, dict):
        return None
    try:
        rate = Decimal(str(fx_quote["rate"]))
    except (KeyError, InvalidOperation, TypeError):
        return None
    base = str(fx_quote.get("base_currency", "")).upper()
    quote_currency = str(fx_quote.get("quote_currency", "")).upper()
    decimal_amount = Decimal(amount) / Decimal(100)
    if currency == base and target == quote_currency:
        converted = decimal_amount * rate
    elif currency == quote_currency and target == base:
        converted = decimal_amount / rate
    else:
        return None  # this quote does not cover this currency pair -- never triangulated
    converted = converted.quantize(Decimal("0.01"), rounding=ROUND_HALF_UP)
    return {"amount_minor_units": int((converted * 100).to_integral_value(rounding=ROUND_HALF_UP)), "currency": target}


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


def format_money_pair(money: Optional[dict[str, Any]], result: Optional[dict[str, Any]]) -> dict[str, Any]:
    """The one shared presentation rule for EVERY rendered price in the
    app (Manual QA remediation Q.1, second correction pass, §1: flight
    cards, stay nightly price, fair-price estimates, stay totals, map
    popups) -- never a per-location reimplementation that could drift.

    Returns {"primary": str, "secondary": Optional[str], "conversion_unavailable": bool}:
    - `primary` is the amount in the trip's own budget currency whenever
      that's determinable (unchanged if already in that currency, or
      genuinely converted using the run's one FX quote) -- this is what
      every USD trip should show as the headline number everywhere.
    - `secondary`, when present, discloses the real raw provider-native
      amount (e.g. the true TRY accommodation price) as detail -- the
      original evidence is never hidden, only demoted to secondary.
    - `conversion_unavailable` is True only when the raw currency
      genuinely differs from the trip's currency and no FX quote could
      convert it -- the caller must show this as an explicit warning and
      MUST NOT label the raw (unconverted) amount with the trip's
      currency."""
    if not isinstance(money, dict) or not money.get("currency"):
        return {"primary": format_money(money), "secondary": None, "conversion_unavailable": False}
    target = get_target_currency(result)
    raw_currency = str(money["currency"]).upper()
    if not target or raw_currency == target:
        return {"primary": format_money(money), "secondary": None, "conversion_unavailable": False}
    normalized = normalize_money(money, target, get_fx_quote(result))
    if normalized is not None:
        return {"primary": format_money(normalized), "secondary": f"Raw: {format_money(money)}", "conversion_unavailable": False}
    return {"primary": format_money(money), "secondary": None, "conversion_unavailable": True}


def nights_from_trip_request(trip_request: Optional[dict[str, Any]]) -> Optional[int]:
    if not isinstance(trip_request, dict) or not trip_request.get("depart_date") or not trip_request.get("return_date"):
        return None
    try:
        from datetime import date as _date
        depart = _date.fromisoformat(str(trip_request["depart_date"]))
        ret = _date.fromisoformat(str(trip_request["return_date"]))
    except ValueError:
        return None
    return max((ret - depart).days, 0)


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
            # Manual QA remediation Q.1 (second correction pass, §1): the
            # raw Money dict, alongside the pre-formatted string below --
            # additive, so any existing reader of `nightly_price` as a
            # string is unaffected; a currency-aware caller uses
            # `nightly_price_raw` with format_money_pair() instead.
            "nightly_price_raw": stay.get("nightly_price"),
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
    # Manual QA remediation Q.1 (§B): prefers the server-computed
    # `budget_summary` (orchestration/system_a/budget_summary.py), which
    # already converts every amount into ONE coherent currency (the
    # trip's own budget currency) using one real, provenance-carrying FX
    # quote -- never mixes a raw TRY accommodation price into a chart
    # labeled with the trip's USD budget currency, which the OLD version
    # of this function did (it took whichever currency the budget field
    # happened to be in and plotted every OTHER price under that same
    # label, regardless of what currency that price actually was).
    budget_summary = result.get("budget_summary") if isinstance(result, dict) else None
    if isinstance(budget_summary, dict):
        return _chart_data_from_budget_summary(budget_summary)

    # Fallback for a result with no budget_summary at all (e.g. a run
    # persisted before this remediation) -- only ever charts amounts
    # already confirmed to share the SAME currency as the budget, never
    # a cross-currency mix.
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
    flight_prices = [
        f["price"]["amount_minor_units"] for f in flights
        if isinstance(f.get("price"), dict) and f["price"].get("amount_minor_units") is not None
        and f["price"].get("currency") == currency
    ]
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
        and s["stay"]["nightly_price"].get("currency") == currency
    ]
    if nightly_prices and nights:
        labels.append("Cheapest stay total")
        values.append(min(nightly_prices) * nights / 100)

    if len(values) < 2:  # need at least a budget + one real cost to be a meaningful comparison
        return None
    return {"labels": labels, "values": values, "currency": currency or ""}


def _chart_data_from_budget_summary(budget_summary: dict[str, Any]) -> Optional[dict[str, list]]:
    budget = budget_summary.get("budget") or {}
    currency = budget.get("currency")
    labels: list[str] = []
    values: list[float] = []
    if budget.get("amount_minor_units") is not None and currency:
        labels.append("Your budget")
        values.append(budget["amount_minor_units"] / 100)

    for key, label in (("cheapest_flight", "Cheapest flight"), ("cheapest_stay_total", "Cheapest stay total")):
        entry = budget_summary.get(key)
        if not isinstance(entry, dict):
            continue
        normalized = entry.get("normalized")
        if isinstance(normalized, dict) and normalized.get("amount_minor_units") is not None:
            labels.append(label)
            values.append(normalized["amount_minor_units"] / 100)

    if len(values) < 2:
        return None
    return {"labels": labels, "values": values, "currency": currency or ""}
