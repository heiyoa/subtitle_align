from __future__ import annotations

from dataclasses import dataclass
from typing import Any, Optional


@dataclass
class RunReceipt:
    started_at_utc: str
    ended_at_utc: Optional[str]
    config: dict[str, Any]
    status: str
    degraded_reason: Optional[str]
    warnings: list[str]
    errors: list[str]
    metrics: dict[str, Any]
    versions: dict[str, Any]

    def to_jsonable(self) -> dict[str, Any]:
        return {
            "started_at_utc": self.started_at_utc,
            "ended_at_utc": self.ended_at_utc,
            "config": self.config,
            "status": self.status,
            "degraded_reason": self.degraded_reason,
            "warnings": self.warnings,
            "errors": self.errors,
            "metrics": self.metrics,
            "versions": self.versions,
        }


def join_degraded_reasons(reasons: list[str]) -> Optional[str]:
    rs = [r for r in reasons if r]
    if not rs:
        return None
    # keep deterministic order while removing duplicates
    seen: set[str] = set()
    out: list[str] = []
    for r in rs:
        if r in seen:
            continue
        seen.add(r)
        out.append(r)
    return ";".join(out)

