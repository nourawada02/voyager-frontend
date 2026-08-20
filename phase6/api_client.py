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
    # Hybrid Chat C.2: the run's own originally-submitted TripRequest
    # (additive field on the existing /v1/runs/{id} response) -- lets the
    # recent-session sidebar restore the full state a later chat
    # modification needs after switching sessions, without a second call.
    trip_request: Optional[dict[str, Any]] = None

    @property
    def is_terminal(self) -> bool:
        return self.status not in ("pending", "running")


@dataclass(frozen=True)
class SSEEvent:
    event_id: str
    stage: str
    payload: dict[str, Any] = field(default_factory=dict)


@dataclass(frozen=True)
class ChatTurnResult:
    session_id: str
    run_id: str
    intent: str
    assistant_message: str
    response_language: str
    trip_patch: Optional[dict[str, Any]]
    requires_new_run: bool
    new_run_id: Optional[str]
    new_run_status: Optional[str]
    clarification_required: bool
    warnings: list[str] = field(default_factory=list)


@dataclass(frozen=True)
class ChatHistoryTurn:
    turn_id: str
    role: str
    content: str
    intent: Optional[str]
    response_language: Optional[str]
    status: str
    created_at: str


@dataclass(frozen=True)
class SessionSummary:
    """Hybrid Chat C.2: one recent-session sidebar row -- a closed, already-safe set of fields only."""

    session_id: str
    latest_run_id: str
    latest_run_status: str
    title: str
    origin: Optional[str]
    destination: Optional[str]
    depart_date: Optional[str]
    return_date: Optional[str]
    preferred_language: Optional[str]
    chat_turn_count: int
    created_at: str
    updated_at: str


_REQUIRED_HEALTH_FIELDS = ("status", "service")
_REQUIRED_RUN_CREATED_FIELDS = ("run_id", "session_id", "status")
_REQUIRED_RUN_STATUS_FIELDS = ("run_id", "session_id", "status", "created_at", "updated_at")
_REQUIRED_CHAT_TURN_FIELDS = (
    "session_id", "run_id", "intent", "assistant_message", "response_language",
    "requires_new_run", "clarification_required",
)
_REQUIRED_CHAT_HISTORY_TURN_FIELDS = ("turn_id", "role", "content", "status", "created_at")
_REQUIRED_SESSION_SUMMARY_FIELDS = (
    "session_id", "latest_run_id", "latest_run_status", "title", "chat_turn_count", "created_at", "updated_at",
)


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
            trip_request=body.get("trip_request"),
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

    def create_chat_turn(
        self, session_id: str, run_id: str, user_message: str, preferred_language: str = "en"
    ) -> ChatTurnResult:
        """Hybrid Chat C.1: the ONLY way `chatbot-ui` ever sends a chat
        message -- always through agent-system-a's own `/v1/chat/turns`,
        never directly to Qwen (this module holds no provider credential
        and never will). Sends only identifiers plus the raw message and
        preferred language -- never a rewritten trip request or result;
        the server loads the authoritative prior state itself."""
        payload = {
            "session_id": session_id, "run_id": run_id,
            "user_message": user_message, "preferred_language": preferred_language,
        }
        response = self._post("/v1/chat/turns", payload)
        if response.status_code != 200:
            raise _parse_error_envelope(response)
        body = response.json()
        _require_fields(body, _REQUIRED_CHAT_TURN_FIELDS, "create_chat_turn")
        return ChatTurnResult(
            session_id=body["session_id"], run_id=body["run_id"], intent=body["intent"],
            assistant_message=body["assistant_message"], response_language=body["response_language"],
            trip_patch=body.get("trip_patch"), requires_new_run=body["requires_new_run"],
            new_run_id=body.get("new_run_id"), new_run_status=body.get("new_run_status"),
            clarification_required=body["clarification_required"], warnings=body.get("warnings") or [],
        )

    def get_chat_history(self, session_id: str, run_id: str) -> list[ChatHistoryTurn]:
        """Persistent-history/grounding correction: restores the full,
        already-safe persisted transcript for one session -- used on
        session load (browser refresh, chatbot-ui restart) to rehydrate
        `st.session_state["chat_messages"]` from System A's own
        persistent store rather than trusting anything client-side."""
        response = self._get(f"/v1/chat/sessions/{session_id}/turns?run_id={run_id}")
        if response.status_code != 200:
            raise _parse_error_envelope(response)
        body = response.json()
        if not isinstance(body, dict) or "turns" not in body:
            raise SystemAResponseError("get_chat_history response missing required field(s): turns")
        turns = []
        for raw in body["turns"]:
            _require_fields(raw, _REQUIRED_CHAT_HISTORY_TURN_FIELDS, "get_chat_history turn")
            turns.append(ChatHistoryTurn(
                turn_id=raw["turn_id"], role=raw["role"], content=raw["content"],
                intent=raw.get("intent"), response_language=raw.get("response_language"),
                status=raw["status"], created_at=raw["created_at"],
            ))
        return turns

    def list_sessions(self, limit: int = 20, offset: int = 0) -> tuple[list[SessionSummary], int]:
        """Hybrid Chat C.2: the recent-session sidebar's one bounded
        listing call -- never one request per sidebar entry. Returns
        `(summaries, total_session_count)`."""
        response = self._get(f"/v1/chat/sessions?limit={limit}&offset={offset}")
        if response.status_code != 200:
            raise _parse_error_envelope(response)
        body = response.json()
        if not isinstance(body, dict) or "sessions" not in body or "total" not in body:
            raise SystemAResponseError("list_sessions response missing required field(s): sessions, total")
        summaries = []
        for raw in body["sessions"]:
            _require_fields(raw, _REQUIRED_SESSION_SUMMARY_FIELDS, "list_sessions session")
            summaries.append(SessionSummary(
                session_id=raw["session_id"], latest_run_id=raw["latest_run_id"],
                latest_run_status=raw["latest_run_status"], title=raw["title"],
                origin=raw.get("origin"), destination=raw.get("destination"),
                depart_date=raw.get("depart_date"), return_date=raw.get("return_date"),
                preferred_language=raw.get("preferred_language"), chat_turn_count=raw["chat_turn_count"],
                created_at=raw["created_at"], updated_at=raw["updated_at"],
            ))
        return summaries, body["total"]

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
