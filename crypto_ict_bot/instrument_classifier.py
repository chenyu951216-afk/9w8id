from __future__ import annotations

from dataclasses import dataclass


QUOTE_SUFFIXES = ("USDT", "USDC", "USD", "PERP")

NON_CRYPTO_PROXY_BASES = {
    "BRENT",
    "CL",
    "GOLD",
    "NGAS",
    "OIL",
    "PAXG",
    "SILVER",
    "UKOIL",
    "USOIL",
    "WTI",
    "XAG",
    "XAU",
    "XAUT",
    "XBR",
    "XNG",
    "XTI",
}

CORE_CRYPTO_BASES = {"BTC", "ETH"}

LARGE_ALT_BASES = {
    "ADA",
    "AVAX",
    "BCH",
    "BNB",
    "DOGE",
    "DOT",
    "LINK",
    "LTC",
    "SOL",
    "SUI",
    "TON",
    "TRX",
    "XRP",
}


@dataclass(frozen=True)
class VolatilityProfile:
    instrument_class: str
    quiet_atr_pct: float
    active_low_atr_pct: float
    active_high_atr_pct: float
    hot_atr_pct: float
    extreme_atr_pct: float
    entry_band_atr_mult: float


@dataclass(frozen=True)
class ParticipationProfile:
    instrument_class: str
    active_low_volume_ratio: float
    active_high_volume_ratio: float
    warm_high_volume_ratio: float
    hot_volume_ratio: float
    extreme_volume_ratio: float


@dataclass(frozen=True)
class TradingStandardProfile:
    instrument_class: str
    min_score_gap: float
    min_htf_context: float
    min_ltf_trigger: float
    min_entry_quality: float
    min_risk_quality: float
    min_rr: float
    scalp_min_rr: float
    strict_min_rr: float
    max_entry_distance_pct: float
    max_entry_band_pct: float
    min_core_data_quality: float
    min_derivatives_context: float
    limit_min_selection_score: float
    limit_min_execution_score: float
    limit_min_setup_score: float
    limit_min_ltf_trigger: float
    limit_min_entry_quality: float
    market_min_execution_score: float
    funding_warm: float
    funding_elevated: float
    funding_extreme: float
    oi_hot_change_pct: float
    oi_extreme_change_pct: float
    crowding_block_score: float
    exhaustion_block_score: float


def symbol_base(symbol: str) -> str:
    base = (symbol or "").upper().strip()
    for suffix in QUOTE_SUFFIXES:
        if base.endswith(suffix) and len(base) > len(suffix):
            return base[: -len(suffix)]
    return base


def is_non_crypto_proxy_symbol(symbol: str) -> bool:
    return symbol_base(symbol) in NON_CRYPTO_PROXY_BASES


def instrument_class(symbol: str) -> str:
    base = symbol_base(symbol)
    if base in NON_CRYPTO_PROXY_BASES:
        return "non_crypto_proxy"
    if base in CORE_CRYPTO_BASES:
        return "core_crypto"
    if base in LARGE_ALT_BASES:
        return "large_altcoin"
    return "altcoin"


def is_altcoin_symbol(symbol: str) -> bool:
    return instrument_class(symbol) in {"large_altcoin", "altcoin"}


def volatility_profile(symbol: str) -> VolatilityProfile:
    kind = instrument_class(symbol)
    if kind == "non_crypto_proxy":
        return VolatilityProfile(
            instrument_class=kind,
            quiet_atr_pct=0.08,
            active_low_atr_pct=0.12,
            active_high_atr_pct=2.2,
            hot_atr_pct=3.4,
            extreme_atr_pct=4.8,
            entry_band_atr_mult=0.32,
        )
    if kind == "core_crypto":
        return VolatilityProfile(
            instrument_class=kind,
            quiet_atr_pct=0.12,
            active_low_atr_pct=0.25,
            active_high_atr_pct=3.5,
            hot_atr_pct=5.0,
            extreme_atr_pct=7.2,
            entry_band_atr_mult=0.32,
        )
    if kind == "large_altcoin":
        return VolatilityProfile(
            instrument_class=kind,
            quiet_atr_pct=0.15,
            active_low_atr_pct=0.35,
            active_high_atr_pct=4.5,
            hot_atr_pct=5.8,
            extreme_atr_pct=8.0,
            entry_band_atr_mult=0.26,
        )
    return VolatilityProfile(
        instrument_class=kind,
        quiet_atr_pct=0.18,
        active_low_atr_pct=0.55,
        active_high_atr_pct=5.4,
        hot_atr_pct=6.6,
        extreme_atr_pct=8.8,
        entry_band_atr_mult=0.22,
    )


def participation_profile(symbol: str) -> ParticipationProfile:
    kind = instrument_class(symbol)
    if kind == "non_crypto_proxy":
        return ParticipationProfile(
            instrument_class=kind,
            active_low_volume_ratio=1.05,
            active_high_volume_ratio=3.2,
            warm_high_volume_ratio=4.5,
            hot_volume_ratio=5.6,
            extreme_volume_ratio=7.2,
        )
    if kind == "core_crypto":
        return ParticipationProfile(
            instrument_class=kind,
            active_low_volume_ratio=1.05,
            active_high_volume_ratio=3.8,
            warm_high_volume_ratio=5.5,
            hot_volume_ratio=7.2,
            extreme_volume_ratio=9.5,
        )
    if kind == "large_altcoin":
        return ParticipationProfile(
            instrument_class=kind,
            active_low_volume_ratio=1.08,
            active_high_volume_ratio=4.6,
            warm_high_volume_ratio=6.4,
            hot_volume_ratio=8.8,
            extreme_volume_ratio=11.0,
        )
    return ParticipationProfile(
        instrument_class=kind,
        active_low_volume_ratio=1.10,
        active_high_volume_ratio=5.2,
        warm_high_volume_ratio=7.0,
        hot_volume_ratio=9.5,
        extreme_volume_ratio=12.0,
    )


def trading_standard_profile(symbol: str) -> TradingStandardProfile:
    kind = instrument_class(symbol)
    if kind == "non_crypto_proxy":
        return TradingStandardProfile(
            instrument_class=kind,
            min_score_gap=8.0,
            min_htf_context=62.0,
            min_ltf_trigger=68.0,
            min_entry_quality=65.0,
            min_risk_quality=62.0,
            min_rr=1.70,
            scalp_min_rr=1.55,
            strict_min_rr=2.00,
            max_entry_distance_pct=0.25,
            max_entry_band_pct=0.60,
            min_core_data_quality=70.0,
            min_derivatives_context=58.0,
            limit_min_selection_score=74.0,
            limit_min_execution_score=65.0,
            limit_min_setup_score=75.0,
            limit_min_ltf_trigger=58.0,
            limit_min_entry_quality=56.0,
            market_min_execution_score=82.0,
            funding_warm=0.00010,
            funding_elevated=0.00025,
            funding_extreme=0.00060,
            oi_hot_change_pct=14.0,
            oi_extreme_change_pct=28.0,
            crowding_block_score=76.0,
            exhaustion_block_score=74.0,
        )
    if kind == "core_crypto":
        return TradingStandardProfile(
            instrument_class=kind,
            min_score_gap=8.0,
            min_htf_context=60.0,
            min_ltf_trigger=65.0,
            min_entry_quality=64.0,
            min_risk_quality=60.0,
            min_rr=1.65,
            scalp_min_rr=1.45,
            strict_min_rr=1.85,
            max_entry_distance_pct=0.30,
            max_entry_band_pct=0.85,
            min_core_data_quality=66.0,
            min_derivatives_context=55.0,
            limit_min_selection_score=72.0,
            limit_min_execution_score=63.0,
            limit_min_setup_score=74.0,
            limit_min_ltf_trigger=55.0,
            limit_min_entry_quality=54.0,
            market_min_execution_score=80.0,
            funding_warm=0.00015,
            funding_elevated=0.00035,
            funding_extreme=0.00080,
            oi_hot_change_pct=18.0,
            oi_extreme_change_pct=35.0,
            crowding_block_score=82.0,
            exhaustion_block_score=82.0,
        )
    if kind == "large_altcoin":
        return TradingStandardProfile(
            instrument_class=kind,
            min_score_gap=8.5,
            min_htf_context=60.0,
            min_ltf_trigger=65.0,
            min_entry_quality=62.0,
            min_risk_quality=60.0,
            min_rr=1.65,
            scalp_min_rr=1.45,
            strict_min_rr=1.95,
            max_entry_distance_pct=0.34,
            max_entry_band_pct=1.20,
            min_core_data_quality=66.0,
            min_derivatives_context=58.0,
            limit_min_selection_score=74.0,
            limit_min_execution_score=65.0,
            limit_min_setup_score=74.0,
            limit_min_ltf_trigger=56.0,
            limit_min_entry_quality=54.0,
            market_min_execution_score=81.0,
            funding_warm=0.00016,
            funding_elevated=0.00040,
            funding_extreme=0.00090,
            oi_hot_change_pct=22.0,
            oi_extreme_change_pct=40.0,
            crowding_block_score=82.0,
            exhaustion_block_score=82.0,
        )
    return TradingStandardProfile(
        instrument_class=kind,
        min_score_gap=9.5,
        min_htf_context=60.0,
        min_ltf_trigger=66.0,
        min_entry_quality=62.0,
        min_risk_quality=60.0,
        min_rr=1.70,
        scalp_min_rr=1.45,
        strict_min_rr=2.05,
        max_entry_distance_pct=0.38,
        max_entry_band_pct=1.65,
        min_core_data_quality=68.0,
        min_derivatives_context=60.0,
        limit_min_selection_score=76.0,
        limit_min_execution_score=68.0,
        limit_min_setup_score=76.0,
        limit_min_ltf_trigger=58.0,
        limit_min_entry_quality=56.0,
        market_min_execution_score=82.0,
        funding_warm=0.00018,
        funding_elevated=0.00045,
        funding_extreme=0.00090,
        oi_hot_change_pct=24.0,
        oi_extreme_change_pct=42.0,
        crowding_block_score=80.0,
        exhaustion_block_score=80.0,
    )


def entry_distance_bands(symbol: str, atr_pct: float | None = None, spread_pct: float | None = None) -> dict[str, float]:
    profile = trading_standard_profile(symbol)
    vol = volatility_profile(symbol)
    atr_component = (atr_pct or 0.0) * vol.entry_band_atr_mult
    spread_component = (spread_pct or 0.0) * 3.0
    execution = max(profile.max_entry_distance_pct, atr_component, spread_component)
    execution = min(execution, profile.max_entry_band_pct)
    return {
        "execution": round(execution, 4),
        "caution": round(max(1.2, execution * 2.4), 4),
        "stale": round(max(3.0, execution * 4.8), 4),
        "missed": round(max(5.0, execution * 7.0), 4),
    }


def volatility_state(symbol: str, atr_pct: float | None) -> str:
    if atr_pct is None:
        return "unknown"
    profile = volatility_profile(symbol)
    if atr_pct < profile.quiet_atr_pct:
        return "quiet"
    if profile.active_low_atr_pct <= atr_pct <= profile.active_high_atr_pct:
        return "active"
    if atr_pct <= profile.hot_atr_pct:
        return "warm"
    if atr_pct < profile.extreme_atr_pct:
        return "hot"
    return "extreme"
