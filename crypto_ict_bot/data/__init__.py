"""Market-data access layer.

The public names are re-exported from the legacy modules so callers can move
to ``crypto_ict_bot.data`` without breaking the current CLI/UI.
"""

from ..exchanges import (  # noqa: F401
    BinanceFuturesClient,
    BybitLinearClient,
    DataUnavailable,
    ExchangeClient,
    Ticker,
    create_auto_client,
    create_client,
)
from ..paid_data import (  # noqa: F401
    check_provider_connections,
    enrich_reports,
    provider_statuses,
)

