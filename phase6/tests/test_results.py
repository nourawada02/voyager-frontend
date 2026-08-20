"""Hermetic tests for phase6.results -- pure functions, no Streamlit, no
network. Fixtures mirror the exact real shapes phase1.models/Travel
MCP/System B produce (verified against orchestration's own D.1/D.2A
tests), never invented shapes.
"""

from __future__ import annotations

from phase6 import results

FLIGHT_ENVELOPE = {
    "schema_version": "1.1.0", "request_id": "r1", "provider": "serpapi-google-flights",
    "data_mode": "live", "status": "success", "retrieved_at": "2026-08-18T00:00:00Z",
    "source_urls": [], "quality": {"schema_version": "1.0.0", "completeness": 1.0, "freshness": "live", "assumptions": []},
    "result": {"options": [
        {"flight_id": "f1", "origin": "BEY", "destination": "IST", "depart_at": "2026-09-10T06:30:00Z",
         "arrive_at": "2026-09-10T08:15:00Z", "carrier": "Turkish Airlines", "stops": 0,
         "price": {"amount_minor_units": 450000, "currency": "TRY"},
         "provenance": {"provider": "serpapi", "data_mode": "live", "retrieved_at": "2026-08-18T00:00:00Z"}},
    ]},
}

STAYS_RESULT = {
    "schema_version": "1.0.0", "snapshot_date": "2026-06-30", "dataset_sha256": "x", "bundle_sha256": "y",
    "model_version": "v1", "currency": "TRY", "data_mode": "historical", "disclaimer": "never live",
    "predictions_produced": True, "ranking_basis": "price_value_desc", "excluded_row_count": 0,
    "stays": [{
        "stay": {
            "stay_id": "stay1", "name": "Boutique Hotel", "district_id": "district_fatih", "side": "european",
            "coordinates": {"lat": 41.0086, "lon": 28.9802}, "nightly_price": {"amount_minor_units": 250000, "currency": "TRY"},
        },
        "fair_price": {
            "schema_version": "1.1.0", "stay_id": "stay1", "estimated_fair_price": {"amount_minor_units": 230000, "currency": "TRY"},
            "scoring_status": "complete", "baseline_beaten": True,
        },
        "rank": 1,
    }],
}

WEATHER_ENVELOPE = {
    "schema_version": "1.0.0", "request_id": "r2", "provider": "open-meteo", "data_mode": "live",
    "retrieved_at": "2026-08-18T00:00:00Z", "source_urls": [],
    "quality": {"schema_version": "1.0.0", "completeness": 1.0, "freshness": "live", "assumptions": []},
    "result": {"location": "Istanbul", "kind": "forecast", "forecast_days": [{"date": "2026-09-10", "condition": "partly_cloudy", "high": 27, "low": 19}]},
}

ITINERARY = {
    "schema_version": "1.0.0", "session_id": "s1", "trace_id": "t1", "contract_version": "1.0.0",
    "recommended_base_candidate_id": "stay1", "accessibility_scores": [{"candidate_id": "stay1", "score": 0.8}],
    "selected_poi_ids": ["poi_hagia_sophia"],
    "daily_plans": [{"schema_version": "1.0.0", "date": "2026-09-10", "side": "european", "poi_ids": ["poi_hagia_sophia"],
                      "legs": [], "walking_minutes": 15.0, "transfer_minutes": 0.0, "activity_minutes": 60.0,
                      "meal_minutes": 0.0, "slack_minutes": 5.0, "warnings": []}],
    "estimated_travel_minutes": 15.0, "expected_walking_minutes": 15.0, "side_crossings": 0,
    "citations": [{"schema_version": "1.0.0", "source_id": "src1", "title": "Hagia Sophia guide"}],
    "assumptions": [], "warnings": ["Qdrant unavailable; degraded to catalog-only scheduling."],
    "data_quality": {"schema_version": "1.0.0", "completeness": 0.9, "freshness": "cached", "assumptions": []},
    "hard_constraint_validation_passed": True,
}

FULL_RESULT = {
    "status": "success",
    "observations": [
        {"action": "get_weather", "status": "success", "fingerprint": "fp1", "envelope": WEATHER_ENVELOPE, "warnings": []},
        {"action": "search_flights", "status": "success", "fingerprint": "fp2", "envelope": FLIGHT_ENVELOPE, "warnings": []},
        {"action": "search_stays", "status": "success", "fingerprint": "fp3", "envelope": STAYS_RESULT, "warnings": []},
        {"action": "call_istanbul_expert", "status": "success", "fingerprint": "fp4", "envelope": ITINERARY, "warnings": []},
    ],
    "warnings": [],
    "narrative": "Here is your plan.",
}

TRIP_REQUEST = {
    "origin": "BEY", "destination": "IST", "depart_date": "2026-09-10", "return_date": "2026-09-15",
    "traveler_count": 2, "budget": {"amount_minor_units": 500000, "currency": "TRY"},
    "preferences": {"interests": ["history"], "pace": "moderate", "language": "en", "mobility_constraints": []},
}


# --- extraction ------------------------------------------------------------------------


def test_extract_flight_options_returns_the_real_options():
    options = results.extract_flight_options(FULL_RESULT)
    assert len(options) == 1
    assert options[0]["flight_id"] == "f1"


def test_extract_stay_items_returns_real_items():
    items = results.extract_stay_items(FULL_RESULT)
    assert len(items) == 1
    assert items[0]["stay"]["stay_id"] == "stay1"


def test_extract_weather_returns_the_unwrapped_result():
    weather = results.extract_weather(FULL_RESULT)
    assert weather["kind"] == "forecast"


def test_extract_itinerary_returns_the_unwrapped_local_itinerary():
    itinerary = results.extract_itinerary(FULL_RESULT)
    assert itinerary["recommended_base_candidate_id"] == "stay1"


def test_extract_fair_price_items_pulls_from_nested_stay_items():
    items = results.extract_fair_price_items(FULL_RESULT)
    assert len(items) == 1
    assert items[0]["estimated_fair_price"]["amount_minor_units"] == 230000


# --- missing optional sections never crash ------------------------------------------------


def test_missing_result_returns_empty_everywhere():
    assert results.extract_flight_options(None) == []
    assert results.extract_stay_items(None) == []
    assert results.extract_weather(None) is None
    assert results.extract_itinerary(None) is None
    assert results.collect_warnings(None) == []
    assert results.collect_citations(None) == []
    assert results.extract_map_points(None) == []
    assert results.build_budget_chart_data(None, None) is None


def test_partial_result_with_only_weather_does_not_crash_other_extractors():
    partial = {"status": "partial", "observations": [
        {"action": "get_weather", "status": "success", "fingerprint": "fp1", "envelope": WEATHER_ENVELOPE, "warnings": []},
    ], "warnings": []}
    assert results.extract_weather(partial) is not None
    assert results.extract_flight_options(partial) == []
    assert results.extract_stay_items(partial) == []
    assert results.extract_itinerary(partial) is None


def test_failed_observation_never_surfaces_as_a_successful_payload():
    failed_result = {"status": "partial", "observations": [
        {"action": "get_weather", "status": "unavailable", "fingerprint": "fp1", "envelope": None, "warnings": ["status=unavailable"]},
    ], "warnings": []}
    assert results.extract_weather(failed_result) is None


# --- extract_weather_status: explains WHY weather is unavailable (Q.1) ---------------


def test_build_budget_chart_data_prefers_server_side_budget_summary():
    """Manual QA remediation Q.1 (§B): when the API result carries a real
    budget_summary (already currency-coherent), the chart is built from
    THAT, never re-derived by naively mixing raw per-observation prices
    of possibly different currencies."""
    result = {
        "status": "success", "observations": [], "warnings": [],
        "budget_summary": {
            "budget": {"amount_minor_units": 500000, "currency": "USD"},
            "fx_quote": {"base_currency": "USD", "quote_currency": "TRY", "rate": "40.00", "effective_date": "2026-08-19", "provider": "frankfurter.app (ECB reference rates)", "retrieved_at": "x", "cache_status": "hit"},
            "fx_status": "success",
            "cheapest_flight": {"raw": {"amount_minor_units": 4409200, "currency": "TRY"}, "normalized": {"amount_minor_units": 110230, "currency": "USD"}},
            "cheapest_stay_total": None,
        },
    }
    chart = results.build_budget_chart_data(result, {"budget": {"amount_minor_units": 500000, "currency": "USD"}})
    assert chart["currency"] == "USD"
    assert chart["labels"] == ["Your budget", "Cheapest flight"]
    assert chart["values"] == [5000.0, 1102.30]


def test_build_budget_chart_data_never_mixes_currencies_without_a_budget_summary():
    """Fallback path (no budget_summary present): a TRY stay price must
    never be charted under a USD budget label."""
    result = {"status": "success", "observations": [
        {"action": "search_stays", "status": "success", "envelope": {"stays": [
            {"stay": {"nightly_price": {"amount_minor_units": 35593, "currency": "TRY"}}},
        ]}},
    ], "warnings": []}
    trip_request = {"depart_date": "2026-09-10", "return_date": "2026-09-13", "budget": {"amount_minor_units": 500000, "currency": "USD"}}
    chart = results.build_budget_chart_data(result, trip_request)
    # Only "Your budget" -- the TRY stay price is correctly excluded, never
    # plotted as if it were USD.
    assert chart is None or "Cheapest stay total" not in chart["labels"]


# --- normalize_money / format_money_pair (Manual QA remediation Q.1, second correction pass §1) ---


_USD_RESULT = {"budget_summary": {"budget": {"amount_minor_units": 500000, "currency": "USD"}, "fx_quote": {
    "base_currency": "USD", "quote_currency": "TRY", "rate": "40.00", "effective_date": "2026-08-19",
    "provider": "frankfurter.app (ECB reference rates)", "retrieved_at": "x", "cache_status": "hit",
}}}
_TRY_RESULT = {"budget_summary": {"budget": {"amount_minor_units": 500000, "currency": "TRY"}, "fx_quote": None}}
_NO_QUOTE_RESULT = {"budget_summary": {"budget": {"amount_minor_units": 500000, "currency": "USD"}, "fx_quote": None, "fx_status": "unavailable"}}


def test_normalize_money_leaves_matching_currency_unchanged():
    assert results.normalize_money({"amount_minor_units": 100, "currency": "TRY"}, "TRY", None) == {"amount_minor_units": 100, "currency": "TRY"}


def test_normalize_money_converts_try_to_usd_decimal_exact():
    quote = {"base_currency": "USD", "quote_currency": "TRY", "rate": "40.00"}
    result = results.normalize_money({"amount_minor_units": 400000, "currency": "TRY"}, "USD", quote)
    assert result == {"amount_minor_units": 10000, "currency": "USD"}  # 4000.00 TRY / 40.00 = 100.00 USD


def test_normalize_money_converts_usd_to_try():
    quote = {"base_currency": "USD", "quote_currency": "TRY", "rate": "40.00"}
    result = results.normalize_money({"amount_minor_units": 10000, "currency": "USD"}, "TRY", quote)
    assert result == {"amount_minor_units": 400000, "currency": "TRY"}


def test_normalize_money_returns_none_without_a_quote():
    assert results.normalize_money({"amount_minor_units": 100, "currency": "TRY"}, "USD", None) is None


def test_normalize_money_returns_none_for_an_unrelated_pair():
    quote = {"base_currency": "USD", "quote_currency": "TRY", "rate": "40.00"}
    assert results.normalize_money({"amount_minor_units": 100, "currency": "EUR"}, "USD", quote) is None


def test_format_money_pair_try_amount_in_try_trip_is_unchanged_and_unflagged():
    pair = results.format_money_pair({"amount_minor_units": 100000, "currency": "TRY"}, _TRY_RESULT)
    assert "TRY" in pair["primary"]
    assert pair["secondary"] is None
    assert pair["conversion_unavailable"] is False


def test_format_money_pair_try_amount_in_usd_trip_is_converted_with_raw_secondary():
    pair = results.format_money_pair({"amount_minor_units": 400000, "currency": "TRY"}, _USD_RESULT)
    assert "USD" in pair["primary"]
    assert "TRY" not in pair["primary"]  # never mislabeled
    assert pair["secondary"] is not None and "TRY" in pair["secondary"]
    assert pair["conversion_unavailable"] is False


def test_format_money_pair_usd_amount_in_usd_trip_needs_no_conversion():
    pair = results.format_money_pair({"amount_minor_units": 10000, "currency": "USD"}, _USD_RESULT)
    assert "USD" in pair["primary"]
    assert pair["secondary"] is None


def test_format_money_pair_conversion_unavailable_shows_raw_currency_never_mislabeled():
    """Required behavior: if FX fails, show native TRY explicitly and
    flag conversion as unavailable -- never label it USD."""
    pair = results.format_money_pair({"amount_minor_units": 400000, "currency": "TRY"}, _NO_QUOTE_RESULT)
    assert "TRY" in pair["primary"]
    assert "USD" not in pair["primary"]
    assert pair["conversion_unavailable"] is True


def test_extract_weather_status_returns_none_when_no_weather_observation_exists():
    assert results.extract_weather_status(None) is None
    assert results.extract_weather_status({"status": "success", "observations": [], "warnings": []}) is None


def test_extract_weather_status_returns_forecast_not_yet_available_with_the_precise_date():
    envelope = {
        "status": "forecast_not_yet_available",
        "result": {"location": "Istanbul", "kind": "unavailable", "earliest_available_forecast_date": "2026-09-04"},
    }
    result = {"status": "partial", "observations": [
        {"action": "get_weather", "status": "forecast_not_yet_available", "fingerprint": "fp1", "envelope": envelope, "warnings": []},
    ], "warnings": []}
    status_info = results.extract_weather_status(result)
    assert status_info["status"] == "forecast_not_yet_available"
    assert status_info["result"]["earliest_available_forecast_date"] == "2026-09-04"


def test_extract_weather_status_returns_the_status_even_for_a_generic_failure():
    result = {"status": "partial", "observations": [
        {"action": "get_weather", "status": "unavailable", "fingerprint": "fp1", "envelope": None, "warnings": ["status=unavailable"]},
    ], "warnings": []}
    status_info = results.extract_weather_status(result)
    assert status_info["status"] == "unavailable"
    assert status_info["result"] is None


# --- warnings / citations / provenance -----------------------------------------------------


def test_collect_warnings_includes_itinerary_level_warnings():
    warnings = results.collect_warnings(FULL_RESULT)
    assert "Qdrant unavailable; degraded to catalog-only scheduling." in warnings


def test_collect_warnings_deduplicates():
    result_with_dupe_warning = {
        "status": "partial",
        "observations": [
            {"action": "get_weather", "status": "unavailable", "fingerprint": "fp1", "envelope": None, "warnings": ["status=unavailable"]},
        ],
        "warnings": ["status=unavailable"],
    }
    warnings = results.collect_warnings(result_with_dupe_warning)
    assert warnings.count("status=unavailable") == 1


def test_collect_citations_returns_real_citations():
    citations = results.collect_citations(FULL_RESULT)
    assert len(citations) == 1
    assert citations[0]["source_id"] == "src1"


def test_collect_itinerary_assumptions_surfaces_the_explicit_fixture_explanation():
    """A real gap found during this checkpoint's own browser smoke test:
    LocalItinerary.assumptions (e.g. "not a real A2A call to System B")
    was always returned by the API but never rendered anywhere -- this
    is the extractor that now feeds the Sources tab's "Assumptions &
    limitations" section."""
    fixture_itinerary = {
        "action": "call_istanbul_expert", "status": "success", "fingerprint": "fp",
        "envelope": {**ITINERARY, "citations": [], "assumptions": ["Deterministic Checkpoint D.0 fixture -- not a real A2A call to System B."]},
        "warnings": [],
    }
    result = {"status": "success", "observations": [fixture_itinerary], "warnings": []}
    assumptions = results.collect_itinerary_assumptions(result)
    assert assumptions == ["Deterministic Checkpoint D.0 fixture -- not a real A2A call to System B."]


def test_collect_itinerary_assumptions_empty_when_no_itinerary():
    assert results.collect_itinerary_assumptions(None) == []
    assert results.collect_itinerary_assumptions({"status": "success", "observations": []}) == []


def test_collect_provenance_badges_one_per_observation():
    badges = results.collect_provenance_badges(FULL_RESULT)
    assert len(badges) == 4
    assert {b["action"] for b in badges} == {"get_weather", "search_flights", "search_stays", "call_istanbul_expert"}


# --- money formatting ------------------------------------------------------------------


def test_format_money_formats_minor_units_correctly():
    assert results.format_money({"amount_minor_units": 450000, "currency": "TRY"}) == "4,500.00 TRY"


def test_format_money_handles_missing_or_malformed_input():
    assert results.format_money(None) == "—"
    assert results.format_money({}) == "—"
    assert results.format_money({"amount_minor_units": None, "currency": "TRY"}) == "—"


# --- map points: only trusted coordinates, never fabricated --------------------------------


def test_extract_map_points_uses_only_real_stay_coordinates():
    points = results.extract_map_points(FULL_RESULT)
    assert len(points) == 1
    assert points[0]["lat"] == 41.0086
    assert points[0]["lon"] == 28.9802
    assert points[0]["label"] == "Boutique Hotel"


def test_extract_map_points_skips_stays_without_coordinates():
    no_coords_result = {
        "status": "success",
        "observations": [{
            "action": "search_stays", "status": "success", "fingerprint": "fp",
            "envelope": {"stays": [{"stay": {"stay_id": "s1", "name": "No Coords Hotel"}, "fair_price": {}, "rank": 1}]},
            "warnings": [],
        }],
        "warnings": [],
    }
    assert results.extract_map_points(no_coords_result) == []


# --- budget chart: only real numeric values, never invented --------------------------------


def test_build_budget_chart_data_uses_real_numbers():
    chart = results.build_budget_chart_data(FULL_RESULT, TRIP_REQUEST)
    assert chart is not None
    assert "Your budget" in chart["labels"]
    assert "Cheapest flight" in chart["labels"]
    assert "Cheapest stay total" in chart["labels"]
    # 5 nights x 2500.00 TRY/night = 12500.00
    stay_index = chart["labels"].index("Cheapest stay total")
    assert chart["values"][stay_index] == 12500.0


def test_build_budget_chart_data_returns_none_with_insufficient_data():
    assert results.build_budget_chart_data({"status": "success", "observations": []}, None) is None


def test_build_budget_chart_data_never_fabricates_a_missing_flight_or_stay_bar():
    weather_only_result = {"status": "partial", "observations": [
        {"action": "get_weather", "status": "success", "fingerprint": "fp1", "envelope": WEATHER_ENVELOPE, "warnings": []},
    ], "warnings": []}
    chart = results.build_budget_chart_data(weather_only_result, TRIP_REQUEST)
    # only "Your budget" would be present -- fewer than 2 values -> None
    assert chart is None
