"""TDD — Tests for risk types: RiskInput and RiskOutput (RED phase).

``RiskInput`` agrupa todo lo que el motor necesita para evaluar una señal.
``RiskOutput`` es el resultado determinista: decisión + parámetros calculados.

Ambos son frozen dataclasses (inmutables) para evitar side-effects.
"""
from decimal import Decimal

import pytest

from trading_system.core.enums import MarketVenue, RiskDecision, SignalAction
from trading_system.risk.types import RiskInput, RiskOutput


class TestRiskInput:
    def test_creation_with_all_fields(self):
        ri = RiskInput(
            signal_action=SignalAction.BUY,
            confidence=Decimal("0.85"),
            asset_symbol="btc_mxn",
            asset_venue=MarketVenue.BITSO,
            current_price=Decimal(500000),
            atr=Decimal(15000),
            portfolio_equity=Decimal(100000),
            current_exposure=Decimal(20000),
            max_capital_pct=Decimal("0.05"),
            max_drawdown_pct=Decimal("0.20"),
            current_drawdown=Decimal("0.05"),
        )
        assert ri.signal_action == SignalAction.BUY
        assert ri.confidence == Decimal("0.85")
        assert ri.asset_symbol == "btc_mxn"

    def test_is_frozen(self):
        ri = RiskInput(
            signal_action=SignalAction.BUY,
            confidence=Decimal("0.8"),
            asset_symbol="btc_mxn",
            asset_venue=MarketVenue.BITSO,
            current_price=Decimal(500000),
            atr=Decimal(15000),
            portfolio_equity=Decimal(100000),
            current_exposure=Decimal(0),
            max_capital_pct=Decimal("0.05"),
            max_drawdown_pct=Decimal("0.20"),
            current_drawdown=Decimal(0),
        )
        with pytest.raises(AttributeError):
            ri.signal_action = SignalAction.SELL  # type: ignore[misc]

    def test_is_market_open_defaults_none(self):
        ri = RiskInput(
            signal_action=SignalAction.BUY,
            confidence=Decimal("0.8"),
            asset_symbol="btc_mxn",
            asset_venue=MarketVenue.BITSO,
            current_price=Decimal(500000),
            atr=None,
            portfolio_equity=Decimal(100000),
            current_exposure=Decimal(0),
            max_capital_pct=Decimal("0.05"),
            max_drawdown_pct=Decimal("0.20"),
            current_drawdown=Decimal(0),
        )
        assert ri.is_market_open is None


class TestRiskOutput:
    def test_approved_output(self):
        ro = RiskOutput(
            decision=RiskDecision.APPROVED,
            reason="All rules passed",
            position_size=Decimal("0.01"),
            stop_loss_price=Decimal(485000),
            take_profit_price=Decimal(515000),
            atr_at_evaluation=Decimal(15000),
            max_capital_pct=Decimal("0.05"),
        )
        assert ro.decision == RiskDecision.APPROVED
        assert ro.position_size == Decimal("0.01")

    def test_rejected_output_has_no_position(self):
        ro = RiskOutput(
            decision=RiskDecision.REJECTED,
            reason="Drawdown exceeded",
            position_size=None,
            stop_loss_price=None,
            take_profit_price=None,
            atr_at_evaluation=Decimal(15000),
            max_capital_pct=None,
        )
        assert ro.decision == RiskDecision.REJECTED
        assert ro.position_size is None

    def test_adjusted_output_has_modified_size(self):
        ro = RiskOutput(
            decision=RiskDecision.ADJUSTED,
            reason="Exposure reduced to stay within limits",
            position_size=Decimal("0.005"),
            stop_loss_price=Decimal(485000),
            take_profit_price=Decimal(515000),
            atr_at_evaluation=Decimal(15000),
            max_capital_pct=Decimal("0.025"),
        )
        assert ro.decision == RiskDecision.ADJUSTED
        assert ro.max_capital_pct == Decimal("0.025")

    def test_is_frozen(self):
        ro = RiskOutput(
            decision=RiskDecision.APPROVED,
            reason="ok",
            position_size=Decimal("0.01"),
            stop_loss_price=Decimal(485000),
            take_profit_price=Decimal(515000),
            atr_at_evaluation=Decimal(15000),
            max_capital_pct=Decimal("0.05"),
        )
        with pytest.raises(AttributeError):
            ro.decision = RiskDecision.REJECTED  # type: ignore[misc]