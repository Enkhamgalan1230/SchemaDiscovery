from __future__ import annotations

from dataclasses import dataclass, field
from typing import Any, Literal, Optional

DecisionType = Literal["PRIMARY_KEY", "FOREIGN_KEY", "COMPOSITE_KEY", "DATATYPE", "INDEX"]
RecStatus = Literal["recommended", "blocked", "info"]
IndexPriority = Literal["HIGH", "MEDIUM", "LOW"]


@dataclass(frozen=True)
class NormalisationDisclosure:
    applied: list[str] = field(default_factory=list)
    notes: str = ""


@dataclass(frozen=True)
class Blocker:
    code: str
    message: str
    metrics: dict[str, Any] = field(default_factory=dict)


@dataclass(frozen=True)
class AltOption:
    # Generic alternative option container
    payload: dict[str, Any]
    confidence: float
    reason_codes: list[str] = field(default_factory=list)
    blockers: list[Blocker] = field(default_factory=list)
    normalisation_disclosure: NormalisationDisclosure = field(default_factory=NormalisationDisclosure)


@dataclass(frozen=True)
class RecommendationRecord:
    decision: DecisionType
    status: RecStatus
    confidence: float
    reason_codes: list[str] = field(default_factory=list)
    blockers: list[Blocker] = field(default_factory=list)
    evidence: dict[str, Any] = field(default_factory=dict)
    normalisation_disclosure: NormalisationDisclosure = field(default_factory=NormalisationDisclosure)
    alternatives: list[AltOption] = field(default_factory=list)

    # Main payload for the recommendation
    payload: dict[str, Any] = field(default_factory=dict)


@dataclass(frozen=True)
class TableRecommendations:
    table_name: str
    row_count: int

    structure: dict[str, Any] = field(default_factory=dict)
    constraints: dict[str, Any] = field(default_factory=dict)
    datatypes: dict[str, Any] = field(default_factory=dict)
    performance: dict[str, Any] = field(default_factory=dict)

    notes: list[str] = field(default_factory=list)


@dataclass(frozen=True)
class SchemaRecommendations:
    meta: dict[str, Any]
    normalisation_policy: dict[str, Any]
    tables: list[TableRecommendations] = field(default_factory=list)
    global_warnings: list[dict[str, Any]] = field(default_factory=list)


def to_dict(obj: Any) -> Any:
    """
    Convert dataclasses to JSON-safe dicts recursively.
    Keep it local to this package to avoid leaking implementation details.
    """
    if obj is None:
        return None

    if isinstance(obj, (str, int, float, bool)):
        return obj

    if isinstance(obj, list):
        return [to_dict(x) for x in obj]

    if isinstance(obj, dict):
        return {str(k): to_dict(v) for k, v in obj.items()}

    # dataclass
    if hasattr(obj, "__dataclass_fields__"):
        out: dict[str, Any] = {}
        for k in obj.__dataclass_fields__.keys():
            out[k] = to_dict(getattr(obj, k))
        return out

    return obj
