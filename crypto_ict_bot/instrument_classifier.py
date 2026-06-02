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
            entry_band_atr_mult=0.35,
        )
    if kind == "large_altcoin":
        return VolatilityProfile(
            instrument_class=kind,
            quiet_atr_pct=0.15,
            active_low_atr_pct=0.35,
            active_high_atr_pct=4.5,
            hot_atr_pct=5.8,
            extreme_atr_pct=8.0,
            entry_band_atr_mult=0.34,
        )
    return VolatilityProfile(
        instrument_class=kind,
        quiet_atr_pct=0.18,
        active_low_atr_pct=0.55,
        active_high_atr_pct=5.4,
        hot_atr_pct=6.6,
        extreme_atr_pct=8.8,
        entry_band_atr_mult=0.33,
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
