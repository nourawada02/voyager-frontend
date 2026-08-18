"""Typed client for the public System A API (Checkpoint Phase 4 D.2A/D.2B).
The ONLY network boundary `chatbot-ui` is ever allowed to cross --
`services/frontend` never imports root packages or another submodule's
code (independent, self-contained submodule, matching CLAUDE.md's
"Repository / submodule boundaries" rule) and never calls providers,
Travel MCP, Qdrant, or System B directly. Every HTTP call in this file
goes to exactly one base URL: `agent-system-a`.

No secret is ever read, held, or logged here -- the public System A API
requires no credential from its caller (auth is explicitly out of scope
for this checkpoint), so there is nothing to redact; this module still
never logs a full response body, only small structured facts, matching
the project-wide "no internal detail reaches a log line unnecessarily"
discipline.
"""

from __future__ import annotations

import json
import os
from dataclasses import dataclass, field
from typing import Any, Iterator, Optional

import httpx

DEFAULT_CONNECT_TIMEOUT_SECONDS = 5.0
DEFAULT_READ_TIMEOUT_SECONDS = 30.0
# SSE connections stay open for the whole run -- bounded by the run's own
# 60s workflow deadline (phase4.models.TOTAL_WORKFLOW_DEADLINE_SECONDS)
# plus headroom, never unbounded.
DEFAULT_SSE_READ_TIMEOUT_SECONDS = 90.0


class SystemAConfigurationError(RuntimeError):
    """Raised when AGENT_SYSTEM_A_BASE_URL is not configured. Never
    guesses an internal hostname -- fails before any network call,
    mirroring this project's existing provider/MCP/A2A client
    configuration-error convention."""


class SystemAConnectionError(RuntimeError):
    """Raised for any transport-level failure (connect timeout, read
    timeout, DNS failure, connection refused, ...). The caller (the
    Streamlit app) renders this as a clear, safe, user-facing message --
    never a raw exception/traceback."""


class SystemAResponseError(RuntimeError):
    """Raised when System A responds, but with an unexpected status code
    or a response body missing a field this client requires -- schema-
    aware parsing rejects a malformed response rather than guessing."""

    def __init__(self, message: str, status_code: Optional[int] = None, error_code: Optional[str] = None):
        super().__init__(message)
        self.status_code = status_code
        self.error_code = error_code


def _base_url(explicit: Optional[str] = None) -> str:
    url = explicit or os.environ.get("AGENT_SYSTEM_A_BASE_URL")
    if not url:
        raise SystemAConfigurationError("AGENT_SYSTEM_A_BASE_URL is required and must be set explicitly")
    return url.rstrip("/")


@dataclass(frozen=True)
class HealthStatus:
    status: str
    service: str
    mode: str


@dataclass(frozen=True)
class RunCreated:
    run_id: str
    session_id: str
    status: str


@dataclass(frozen=True)
class RunStatus:
    run_id: str
    session_id: str
    status: str
    created_at: str
    updated_at: str
    result: Optional[dict[str, Any]] = None

    @property
    def is_terminal(self) -> bool:
        return self.status not in ("pending", "running")


@dataclass(frozen=True)
class SSEEvent:
    event_id: str
    stage: str
    payload: dict[str, Any] = field(default_factory=dict)


_REQUIRED_HEALTH_FIELDS = ("status", "service")
_REQUIRED_RUN_CREATED_FIELDS = ("run_id", "session_id", "status")
_REQUIRED_RUN_STATUS_FIELDS = ("run_id", "session_id", "status", "created_at", "updated_at")


def _parse_error_envelope(response: httpx.Response) -> SystemAResponseError:
    try:
        body = response.json()
    except (json.JSONDecodeError, ValueError):
        return SystemAResponseError(f"System A returned status {response.status_code}", status_code=response.status_code)
    message = body.get("message") if isinstance(body, dict) else None
    error_code = body.get("error_code") if isinstance(body, dict) else None
    return SystemAResponseError(
        message or f"System A returned status {response.status_code}",
        status_code=response.status_code, error_code=error_code,
    )


def _require_fields(body: Any, fields: tuple[str, ...], what: str) -> None:
    if not isinstance(body, dict):
        raise SystemAResponseError(f"{what} response was not a JSON object")
    missing = [f for f in fields if f not in body]
    if missing:
        raise SystemAResponseError(f"{what} response missing required field(s): {', '.join(missing)}")


class SystemAClient:
    """One instance per Streamlit session is fine -- this class holds no
    mutable network state itself (httpx opens/closes a connection per
    call), only configuration."""

    def __init__(
        self,
        base_url: Optional[str] = None,
        connect_timeout: float = DEFAULT_CONNECT_TIMEOUT_SECONDS,
        read_timeout: float = DEFAULT_READ_TIMEOUT_SECONDS,
    ):
        self.base_url = _base_url(base_url)
        self._timeout = httpx.Timeout(connect=connect_timeout, read=read_timeout, write=connect_timeout, pool=connect_timeout)
        self._sse_timeout = httpx.Timeout(connect=connect_timeout, read=DEFAULT_SSE_READ_TIMEOUT_SECONDS, write=connect_timeout, pool=connect_timeout)

    def _get(self, path: str) -> httpx.Response:
        try:
            return httpx.get(f"{self.base_url}{path}", timeout=self._timeout)
        except httpx.HTTPError as exc:
            raise SystemAConnectionError(f"Could not reach agent-system-a at {self.base_url}") from exc

    def _post(self, path: str, json_body: Optional[dict] = None) -> httpx.Response:
        try:
            return httpx.post(f"{self.base_url}{path}", json=json_body, timeout=self._timeout)
        except httpx.HTTPError as exc:
            raise SystemAConnectionError(f"Could not reach agent-system-a at {self.base_url}") from exc

    def health(self) -> HealthStatus:
        response = self._get("/health")
        if response.status_code != 200:
            raise _parse_error_envelope(response)
        body = response.json()
        _require_fields(body, _REQUIRED_HEALTH_FIELDS, "health")
        return HealthStatus(status=body["status"], service=body["service"], mode=body.get("mode", "unknown"))

    def create_run(
        self, user_message: str, trip_request: Optional[dict[str, Any]] = None, idempotency_key: Optional[str] = None
    ) -> RunCreated:
        payload: dict[str, Any] = {"user_message": user_message}
        if trip_request is not None:
            payload["trip_request"] = trip_request
        if idempotency_key is not None:
            payload["idempotency_key"] = idempotency_key
        response = self._post("/v1/runs", payload)
        if response.status_code not in (200, 201):
            raise _parse_error_envelope(response)
        body = response.json()
        _require_fields(body, _REQUIRED_RUN_CREATED_FIELDS, "create_run")
        return RunCreated(run_id=body["run_id"], session_id=body["session_id"], status=body["status"])

    def get_run(self, run_id: str) -> RunStatus:
        response = self._get(f"/v1/runs/{run_id}")
        if response.status_code != 200:
            raise _parse_error_envelope(response)
        body = response.json()
        _require_fields(body, _REQUIRED_RUN_STATUS_FIELDS, "get_run")
        return RunStatus(
            run_id=body["run_id"], session_id=body["session_id"], status=body["status"],
            created_at=body["created_at"], updated_at=body["updated_at"], result=body.get("result"),
        )

    def cancel_run(self, run_id: str) -> RunStatus:
        response = self._post(f"/v1/runs/{run_id}/cancel")
        if response.status_code != 200:
            raise _parse_error_envelope(response)
        body = response.json()
        _require_fields(body, _REQUIRED_RUN_STATUS_FIELDS, "cancel_run")
        return RunStatus(
            run_id=body["run_id"], session_id=body["session_id"], status=body["status"],
            created_at=body["created_at"], updated_at=body["updated_at"], result=body.get("result"),
        )

    def stream_events(self, run_id: str, last_event_id: Optional[str] = None) -> Iterator[SSEEvent]:
        """Yields events in order, terminating naturally when the server
        sends a terminal-stage event or closes the stream. Supports
        reconnect/replay via `last_event_id` (sent as the `Last-Event-ID`
        header, exactly what the server already supports)."""
        headers = {}
        if last_event_id:
            headers["Last-Event-ID"] = last_event_id
        try:
            with httpx.stream(
                "GET", f"{self.base_url}/v1/runs/{run_id}/events", headers=headers, timeout=self._sse_timeout
            ) as response:
                if response.status_code != 200:
                    response.read()
                    raise _parse_error_envelope(response)
                yield from _parse_sse_lines(response.iter_lines())
        except httpx.HTTPError as exc:
            raise SystemAConnectionError(f"Could not reach agent-system-a at {self.base_url}") from exc


def _parse_sse_lines(lines: Iterator[str]) -> Iterator[SSEEvent]:
    """A small, local, spec-literal SSE parser (never a third-party SSE
    client library) -- parses exactly the wire format
    `orchestration.system_a.sse.format_event` produces: `id:`/`event:`/
    `data:` lines, blank-line-terminated, `:`-prefixed heartbeat comments
    ignored."""
    event_id: Optional[str] = None
    event_name: Optional[str] = None
    data_lines: list[str] = []

    for line in lines:
        if line.startswith(":"):
            continue  # heartbeat/comment
        if line == "":
            if event_name is not None:
                raw_data = "\n".join(data_lines)
                try:
                    payload = json.loads(raw_data) if raw_data else {}
                except json.JSONDecodeError:
                    payload = {}
                yield SSEEvent(event_id=event_id or "", stage=event_name, payload=payload if isinstance(payload, dict) else {})
            event_id, event_name, data_lines = None, None, []
            continue
        if line.startswith("id: "):
            event_id = line[len("id: "):]
        elif line.startswith("event: "):
            event_name = line[len("event: "):]
        elif line.startswith("data: "):
            data_lines.append(line[len("data: "):])
