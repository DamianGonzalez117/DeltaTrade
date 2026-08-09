"""TDD — Tests for RiskRule ABC and all concrete rules (RED phase).

Cada regla implementa ``RiskRule.evaluate(input) -> RuleResult``:
  - RuleResult trae: passed (bool), reason (str), y ajustes opcionales
    (position_size_pct, stop_loss_price, take_profit_price).

El motor ejecuta las reglas en orden y combina los resultados.
"""

from decimal import Decimal

import pytest

from trading_system.core.enums import MarketVenue, RiskDecision, SignalAction
from trading_system.risk.engine import RiskEngine
from trading_system.risk.rules.base import RiskRule, RuleResult
from trading_system.risk.rules.drawdown import DrawdownRule
from trading_system.risk.rules.exposure import MaxExposureRule
from trading_system.risk.rules.hours import MarketHoursRule
from trading_system.risk.rules.sizing import VolatilityPositionSizer
from trading_system.risk.types import RiskInput

# ---------------------------------------------------------------------------
# Fixtures
# ---------------------------------------------------------------------------

def _make_input(**overrides) -> RiskInput:
    defaults = {
        "signal_action": SignalAction.BUY,
        "confidence": Decimal("0.85"),
        "asset_symbol": "btc_mxn",
        "asset_venue": MarketVenue.BITSO,
        "current_price": Decimal(500000),
        "atr": Decimal(15000),
        "portfolio_equity": Decimal(100000),
        "current_exposure": Decimal(20000),
        "max_capital_pct": Decimal("0.05"),
        "max_drawdown_pct": Decimal("0.20"),
        "current_drawdown": Decimal("0.05"),
    }
    defaults.update(overrides)
    return RiskInput(**defaults)


# ---------------------------------------------------------------------------
# RiskRule ABC
# ---------------------------------------------------------------------------

class TestRiskRuleABC:
    def test_cannot_instantiate_directly(self):
        with pytest.raises(TypeError):
            RiskRule()  # type: ignore[abstract]


# ---------------------------------------------------------------------------
# MaxExposureRule
# ---------------------------------------------------------------------------

class TestMaxExposureRule:
    def test_passes_when_exposure_within_limits(self):
        rule = MaxExposureRule()
        result = rule.evaluate(_make_input(current_exposure=Decimal(20000)))
        assert result.passed is True

    def test_rejects_when_exposure_exceeds_max(self):
        rule = MaxExposureRule(max_total_exposure_pct=Decimal("0.30"))
        result = rule.evaluate(_make_input(
            current_exposure=Decimal(40000),
            portfolio_equity=Decimal(100000),
        ))
        assert result.passed is False
        assert "exposure" in result.reason.lower()

    def test_hold_signal_always_passes(self):
        rule = MaxExposureRule(max_total_exposure_pct=Decimal("0.01"))
        result = rule.evaluate(_make_input(
            signal_action=SignalAction.HOLD,
            current_exposure=Decimal(90000),
        ))
        assert result.passed is True


# ---------------------------------------------------------------------------
# DrawdownRule
# ---------------------------------------------------------------------------

class TestDrawdownRule:
    def test_passes_when_drawdown_within_limits(self):
        rule = DrawdownRule()
        result = rule.evaluate(_make_input(current_drawdown=Decimal("0.05")))
        assert result.passed is True

    def test_rejects_when_drawdown_exceeds_max(self):
        rule = DrawdownRule()
        result = rule.evaluate(_make_input(
            current_drawdown=Decimal("0.25"),
            max_drawdown_pct=Decimal("0.20"),
        ))
        assert result.passed is False
        assert "drawdown" in result.reason.lower()

    def test_hold_signal_passes_even_in_drawdown(self):
        rule = DrawdownRule()
        result = rule.evaluate(_make_input(
            signal_action=SignalAction.HOLD,
            current_drawdown=Decimal("0.30"),
        ))
        assert result.passed is True


# ---------------------------------------------------------------------------
# VolatilityPositionSizer
# ---------------------------------------------------------------------------

class TestVolatilityPositionSizer:
    def test_calculates_position_size_based_on_atr(self):
        sizer = VolatilityPositionSizer(atr_multiplier=Decimal("1.5"))
        result = sizer.evaluate(_make_input(
            atr=Decimal(15000),
            current_price=Decimal(500000),
            portfolio_equity=Decimal(100000),
            max_capital_pct=Decimal("0.05"),
        ))
        assert result.passed is True
        assert result.position_size_pct is not None
        assert result.position_size_pct > 0

    def test_returns_none_size_when_no_atr(self):
        sizer = VolatilityPositionSizer()
        result = sizer.evaluate(_make_input(atr=None))
        assert result.passed is True
        assert result.position_size_pct is None

    def test_hold_signal_skips_sizing(self):
        sizer = VolatilityPositionSizer()
        result = sizer.evaluate(_make_input(signal_action=SignalAction.HOLD))
        assert result.passed is True
        assert result.position_size_pct is None

    def test_size_capped_at_max_capital_pct(self):
        sizer = VolatilityPositionSizer(atr_multiplier=Decimal("1.0"))
        # ATR tiny → position would be huge, but capped
        result = sizer.evaluate(_make_input(
            atr=Decimal(10),
            current_price=Decimal(500000),
            portfolio_equity=Decimal(100000),
            max_capital_pct=Decimal("0.05"),
        ))
        assert result.position_size_pct is not None
        assert result.position_size_pct <= Decimal("0.05")

    def test_provides_stop_and_take_profit(self):
        sizer = VolatilityPositionSizer(
            atr_multiplier=Decimal("1.5"),
            sl_atr_mult=Decimal("2.0"),
            tp_atr_mult=Decimal("3.0"),
        )
        result = sizer.evaluate(_make_input(
            signal_action=SignalAction.BUY,
            atr=Decimal(15000),
            current_price=Decimal(500000),
            portfolio_equity=Decimal(100000),
            max_capital_pct=Decimal("0.05"),
        ))
        assert result.stop_loss_price == Decimal(470000)  # 500000 - 2*15000
        assert result.take_profit_price == Decimal(545000)  # 500000 + 3*15000

    def test_sell_signal_stop_and_take_profit_inverted(self):
        sizer = VolatilityPositionSizer(
            sl_atr_mult=Decimal("2.0"),
            tp_atr_mult=Decimal("3.0"),
        )
        result = sizer.evaluate(_make_input(
            signal_action=SignalAction.SELL,
            atr=Decimal(15000),
            current_price=Decimal(500000),
            portfolio_equity=Decimal(100000),
            max_capital_pct=Decimal("0.05"),
        ))
        assert result.stop_loss_price == Decimal(530000)  # 500000 + 2*15000
        assert result.take_profit_price == Decimal(455000)  # 500000 - 3*15000


# ---------------------------------------------------------------------------
# MarketHoursRule
# ---------------------------------------------------------------------------

class TestMarketHoursRule:
    def test_passes_when_market_open(self):
        rule = MarketHoursRule()
        result = rule.evaluate(_make_input(is_market_open=True))
        assert result.passed is True

    def test_rejects_when_market_closed(self):
        rule = MarketHoursRule()
        result = rule.evaluate(_make_input(is_market_open=False))
        assert result.passed is False
        assert "market" in result.reason.lower()

    def test_skips_check_when_unknown(self):
        rule = MarketHoursRule()
        result = rule.evaluate(_make_input(is_market_open=None))
        assert result.passed is True

    def test_hold_passes_even_when_closed(self):
        rule = MarketHoursRule()
        result = rule.evaluate(_make_input(
            signal_action=SignalAction.HOLD,
            is_market_open=False,
        ))
        assert result.passed is True


# ---------------------------------------------------------------------------
# RiskEngine
# ---------------------------------------------------------------------------

class TestRiskEngine:
    def test_all_rules_pass_returns_approved(self):
        engine = RiskEngine.create_default()
        result = engine.evaluate(_make_input(
            signal_action=SignalAction.BUY,
            current_exposure=Decimal(10000),
            current_drawdown=Decimal("0.02"),
            is_market_open=True,
            atr=Decimal(15000),
        ))
        assert result.decision == RiskDecision.APPROVED
        assert result.position_size is not None
        assert result.stop_loss_price is not None
        assert result.take_profit_price is not None
        assert result.atr_at_evaluation == Decimal(15000)

    def test_drawdown_exceeded_returns_rejected(self):
        engine = RiskEngine.create_default()
        result = engine.evaluate(_make_input(
            current_drawdown=Decimal("0.25"),
            max_drawdown_pct=Decimal("0.20"),
            is_market_open=True,
        ))
        assert result.decision == RiskDecision.REJECTED
        assert result.position_size is None
        assert "drawdown" in result.reason.lower()

    def test_market_closed_returns_rejected(self):
        engine = RiskEngine.create_default()
        result = engine.evaluate(_make_input(
            is_market_open=False,
            current_drawdown=Decimal("0.02"),
            current_exposure=Decimal(5000),
        ))
        assert result.decision == RiskDecision.REJECTED
        assert "market" in result.reason.lower()

    def test_hold_signal_returned_as_approved_no_size(self):
        engine = RiskEngine.create_default()
        result = engine.evaluate(_make_input(
            signal_action=SignalAction.HOLD,
            is_market_open=True,
        ))
        assert result.decision == RiskDecision.APPROVED
        assert result.position_size is None or result.position_size == Decimal(0)

    def test_exposure_exceeded_rejects(self):
        engine = RiskEngine.create_default()
        result = engine.evaluate(_make_input(
            current_exposure=Decimal(60000),
            portfolio_equity=Decimal(100000),
            is_market_open=True,
            current_drawdown=Decimal("0.02"),
        ))
        assert result.decision == RiskDecision.REJECTED
        assert "exposure" in result.reason.lower()

    def test_reason_concatenates_all_failed_rules(self):
        engine = RiskEngine.create_default()
        result = engine.evaluate(_make_input(
            is_market_open=False,
            current_drawdown=Decimal("0.30"),
            current_exposure=Decimal(90000),
        ))
        assert result.decision == RiskDecision.REJECTED
        assert "drawdown" in result.reason.lower()
        assert "exposure" in result.reason.lower()
        assert "market" in result.reason.lower()

    def test_custom_rules_injected(self):
        class AlwaysPass(RiskRule):
            @property
            def name(self) -> str:
                return "always_pass"

            def evaluate(self, inp: RiskInput) -> RuleResult:
                return RuleResult(passed=True, reason="ok")

        engine = RiskEngine(rules=[AlwaysPass()])
        result = engine.evaluate(_make_input())
        assert result.decision == RiskDecision.APPROVED