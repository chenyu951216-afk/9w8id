from __future__ import annotations

from dataclasses import dataclass, field
from datetime import datetime, timezone
from typing import Any


@dataclass(frozen=True)
class Candle:
    open_time: datetime
    open: float
    high: float
    low: float
    close: float
    volume: float
    close_time: datetime | None = None

    @property
    def ts_ms(self) -> int:
        return int(self.open_time.timestamp() * 1000)

    @property
    def body(self) -> float:
        return abs(self.close - self.open)

    @property
    def range(self) -> float:
        return max(self.high - self.low, 0.0)

    @property
    def direction(self) -> str:
        if self.close > self.open:
            return "bull"
        if self.close < self.open:
            return "bear"
        return "flat"


@dataclass(frozen=True)
class Swing:
    kind: str
    index: int
    time: datetime
    price: float


@dataclass(frozen=True)
class FVG:
    direction: str
    index: int
    start_time: datetime
    lower: float
    upper: float
    size_pct: float
    tapped: bool = False
    filled: bool = False
    overlap_order_block: bool = False

    @property
    def midpoint(self) -> float:
        return (self.lower + self.upper) / 2


@dataclass(frozen=True)
class Sweep:
    direction: str
    index: int
    time: datetime
    level: float
    extreme: float
    strength: float


@dataclass(frozen=True)
class StructureBreak:
    direction: str
    index: int
    time: datetime
    level: float
    close: float
    kind: str


@dataclass(frozen=True)
class Displacement:
    direction: str
    index: int
    time: datetime
    body_atr: float
    close_location: float
    has_fvg: bool


@dataclass(frozen=True)
class Zone:
    direction: str
    kind: str
    lower: float
    upper: float
    index: int
    time: datetime

    @property
    def midpoint(self) -> float:
        return (self.lower + self.upper) / 2


@dataclass
class DirectionScore:
    direction: str
    score: float = 0.0
    max_score: float = 0.0
    reference_max_score: float = 100.0
    reasons: list[str] = field(default_factory=list)
    warnings: list[str] = field(default_factory=list)
    entry_zone: tuple[float, float] | None = None
    stop: float | None = None
    target: float | None = None
    take_profits: list[dict[str, Any]] = field(default_factory=list)
    rr: float | None = None
    feature_scores: dict[str, float] = field(default_factory=dict)
    feature_max_scores: dict[str, float] = field(default_factory=dict)
    skipped_features: dict[str, str] = field(default_factory=dict)

    @property
    def normalized(self) -> float:
        if self.max_score <= 0:
            return 0.0
        return max(0.0, min(100.0, self.score / self.max_score * 100.0))

    @property
    def data_completeness(self) -> float:
        if self.reference_max_score <= 0:
            return 0.0
        return max(0.0, min(100.0, self.max_score / self.reference_max_score * 100.0))


@dataclass
class SymbolReport:
    symbol: str
    exchange: str
    price: float
    quote_volume_24h: float
    change_pct_24h: float
    data_time: datetime
    selected_direction: str
    score: float
    long: DirectionScore
    short: DirectionScore
    data_coverage: dict[str, int]
    missing_data: list[str] = field(default_factory=list)
    metadata: dict[str, Any] = field(default_factory=dict)

    @property
    def grade(self) -> str:
        if self.score >= 82:
            return "A"
        if self.score >= 72:
            return "B"
        if self.score >= 62:
            return "C"
        if self.score >= 52:
            return "Watch"
        return "Skip"


def utc_from_ms(value: int | str | float) -> datetime:
    return datetime.fromtimestamp(int(value) / 1000, tz=timezone.utc)
