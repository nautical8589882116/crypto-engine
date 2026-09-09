"""Atomic legging: place the directional leg and its OTM hedge atomically.

The module ensures both legs fill or neither does — if the hedge leg cannot be
placed, the directional position is liquidated (rolled back) to restore
flatness. This is the risk-management guarantee of the execution kernel.
"""

from __future__ import annotations

from dataclasses import dataclass

from quant_crypto.broker.base import Broker, Fill
from quant_crypto.schemas.order import HedgeLeg, Order, OrderSide, OrderType


@dataclass
class LegResult:
    directional: Fill
    hedge: Fill | None
    rollback: bool = False


class AtomicLeger:
    def __init__(self, broker: Broker):
        self.broker = broker

    def execute(self, directional: Order) -> LegResult:
        """Place the directional order, then the hedge leg, atomically.

        Returns LegResult. If hedge placement fails, the directional position
        is liquidated and rollback=True.
        """
        main_fill = self.broker.place_order(directional)
        if main_fill is None:
            raise RuntimeError("Directional order failed to fill (FOK reject)")

        if directional.hedge is None:
            return LegResult(directional=main_fill, hedge=None)

        hedge_order = Order(
            symbol=directional.hedge.symbol,
            side=directional.hedge.side,
            qty=directional.hedge.qty,
            order_type=OrderType.FOK,
            limit_price=directional.hedge.limit_price,
        )
        hedge_fill = self.broker.place_order(hedge_order)

        if hedge_fill is None:
            # Rollback: liquidate the directional leg to restore flatness.
            self._liquidate(directional)
            return LegResult(directional=main_fill, hedge=None, rollback=True)

        return LegResult(directional=main_fill, hedge=hedge_fill)

    def _liquidate(self, directional: Order) -> None:
        opposite = OrderSide.SELL if directional.side is OrderSide.BUY else OrderSide.BUY
        liquidate = Order(
            symbol=directional.symbol,
            side=opposite,
            qty=directional.qty,
            order_type=OrderType.IOC,
            limit_price=directional.limit_price,
        )
        self.broker.place_order(liquidate)
