"""Structured query builder and response synthesizer strictly adhering to Rule 2."""

from __future__ import annotations

from dataclasses import dataclass
from typing import Any

from cctv.db import NO_EVIDENCE_MESSAGE, EventDatabase


@dataclass
class EventQuery:
    """Structured search query for event filtering."""

    camera_id: str | None = None
    event_type: str | None = None
    class_name: str | None = None
    zone_name: str | None = None
    start_time: float | None = None
    end_time: float | None = None
    min_det_conf: float | None = None


@dataclass
class QueryResult:
    """Standardized response from the search engine."""

    found: bool
    message: str
    events: list[dict[str, Any]]


class EventQueryEngine:
    """Executes structured queries against the event database without fabrication (Rule 2)."""

    def __init__(self, db: EventDatabase) -> None:
        self.db = db

    def execute(self, query: EventQuery) -> QueryResult:
        """Execute query. If no records match, return exact no-evidence text per Rule 2."""
        rows = self.db.search_events(
            camera_id=query.camera_id,
            event_type=query.event_type,
            class_name=query.class_name,
            zone_name=query.zone_name,
            start_time=query.start_time,
            end_time=query.end_time,
            min_det_conf=query.min_det_conf,
        )

        if not rows:
            return QueryResult(
                found=False,
                message=NO_EVIDENCE_MESSAGE,
                events=[],
            )

        return QueryResult(
            found=True,
            message=f"Found {len(rows)} matching event(s).",
            events=rows,
        )
