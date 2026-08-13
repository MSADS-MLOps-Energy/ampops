"""Request/response models for the serving API.

Timestamps are ISO-8601 **naive** strings on the fixed UTC-5 grid — the same
convention as `joined_hourly.parquet`'s `time` column. Tz-aware input is
rejected rather than converted: the grid is an internal invariant, and silently
shifting a client's `-06:00` timestamp onto it would return a confidently wrong
hour instead of an error the caller can see.

`protected_namespaces=()` is set wherever a field starts with `model_`. Those
names come from the contract's response shapes (§5), and pydantic v2 otherwise
warns about them on every import.
"""

from __future__ import annotations

from datetime import datetime
from typing import Literal

from pydantic import BaseModel, ConfigDict, Field, field_validator

_TZ_ERROR = "timestamps must be naive ISO-8601 on the fixed UTC-5 grid, not tz-aware"


class PredictRequest(BaseModel):
    grid_id: str
    timestamp: datetime

    @field_validator("timestamp")
    @classmethod
    def _must_be_naive(cls, value: datetime) -> datetime:
        if value.tzinfo is not None:
            raise ValueError(_TZ_ERROR)
        return value


class PredictResponse(BaseModel):
    model_config = ConfigDict(protected_namespaces=())

    grid_id: str
    timestamp: datetime
    predicted_mw: float
    model_version: str
    source: Literal["cache", "live"]
    latency_ms: float


class BatchPredictRequest(BaseModel):
    grid_id: str
    timestamps: list[datetime] = Field(min_length=1)

    @field_validator("timestamps")
    @classmethod
    def _must_all_be_naive(cls, values: list[datetime]) -> list[datetime]:
        if any(value.tzinfo is not None for value in values):
            raise ValueError(_TZ_ERROR)
        return values


class Prediction(BaseModel):
    timestamp: datetime
    predicted_mw: float


class BatchPredictResponse(BaseModel):
    """Shared by `/predict/batch` and `/forecast` so a consumer parses one shape."""

    model_config = ConfigDict(protected_namespaces=())

    grid_id: str
    model_version: str
    predictions: list[Prediction]


class HealthResponse(BaseModel):
    status: str


class ReadinessResponse(BaseModel):
    """`/ready`'s body names the failing check — that is its diagnostic value."""

    model_config = ConfigDict(protected_namespaces=())

    ready: bool
    model_loaded: bool
    h2o_cluster: bool
    feature_store: bool
    model_uri: str | None = None
    model_version: str | None = None
    detail: str | None = None
