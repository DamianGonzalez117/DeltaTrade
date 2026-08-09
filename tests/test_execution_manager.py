"""TDD — Tests for ExecutionManager (RED phase).

``ExecutionManager`` orquesta el pipeline completo:
  1. HybridInferenceEngine.infer(InferenceInput) → InferenceOutput
  2. Construye RiskInput a partir de InferenceOutput + portfolio state
  3. RiskEngine.evaluate(RiskInput) → RiskOutput
  4. OrderExecutor.execute(RiskOutput, ...) → ExecutionResult | None

DIP: todas las dependencias se inyectan (inference, risk, executor, portfolio provider).
"""
from decimal import Decimal
from unittest.mock import AsyncMock, MagicMock

from trading_system.connectors.base import ExecutionResult
from trading_system.core.enums import (
    AssetClass,
    MarketVenue,
    RiskDecision,
    SignalAction,
)
from trading_system.execution.manager import ExecutionManager
from trading_system.inference.types import InferenceInput, InferenceOutput
from trading_system.risk.types import RiskInput, RiskOutput


def _mock_inference_engine() -> MagicMock:
    engine = MagicMock()
    engine.infer = MagicMock(
        return_value=InferenceOutput(
            action=SignalAction.BUY,
            confidence=Decimal("0.85"),
            model_name="hybrid_v1",
            model_version="0.1.0",
            technical_score=Decimal("0.75"),
            sentiment_score=Decimal("0.80"),
            technical_weight=Decimal("0.6"),
            sentiment_weight=Decimal("0.4"),
        )
    )
    return engine


def _mock_risk_engine() -> MagicMock:
    engine = MagicMock()
    engine.evaluate = MagicMock(
        return_value=RiskOutput(
            decision=RiskDecision.APPROVED,
            reason="All rules passed",
            position_size=Decimal("0.01"),
            stop_loss_price=Decimal(485000),
            take_profit_price=Decimal(515000),
            atr_at_evaluation=Decimal(15000),
            max_capital_pct=Decimal("0.05"),
        )
    )
    return engine


def _mock_executor() -> AsyncMock:
    executor = AsyncMock()
    executor.execute = AsyncMock(
        return_value=ExecutionResult(
            external_order_id="ex-123",
            status="open",
            filled_quantity=Decimal(0),
            average_price=None,
            raw_response={"success": True},
        )
    )
    return executor


def _mock_portfolio_provider() -> MagicMock:
    provider = MagicMock()
    provider.get_portfolio_state = MagicMock(
        return_value={
            "equity": Decimal(100000),
            "current_exposure": Decimal(20000),
            "current_drawdown": Decimal("0.05"),
            "max_capital_pct": Decimal("0.05"),
            "max_drawdown_pct": Decimal("0.20"),
            "current_price": Decimal(500000),
            "atr": Decimal(15000),
            "is_market_open": True,
        }
    )
    return provider


def _make_inference_input() -> InferenceInput:
    return InferenceInput(
        asset_symbol="btc_mxn",
        asset_venue=MarketVenue.BITSO,
        asset_class=AssetClass.CRYPTO,
        technical_features={"rsi_14": 65.0, "atr_14": 15000.0},
        headlines=["Bitcoin rally continues"],
    )


class TestRunFullPipeline:
    async def test_buy_signal_approved_and_executed(self):
        manager = ExecutionManager(
            inference_engine=_mock_inference_engine(),
            risk_engine=_mock_risk_engine(),
            executor=_mock_executor(),
            portfolio_provider=_mock_portfolio_provider(),
        )

        result = await manager.run(
            inference_input=_make_inference_input(),
        )

        assert result is not None
        assert result.external_order_id == "ex-123"

    async def test_hold_signal_does_not_execute(self):
        inf_engine = _mock_inference_engine()
        inf_engine.infer = MagicMock(
            return_value=InferenceOutput(
                action=SignalAction.HOLD,
                confidence=Decimal("0.40"),
                model_name="hybrid_v1",
                model_version="0.1.0",
                technical_score=Decimal("0.0"),
                sentiment_score=Decimal("0.0"),
                technical_weight=Decimal("0.6"),
                sentiment_weight=Decimal("0.4"),
            )
        )
        risk_engine = _mock_risk_engine()
        risk_engine.evaluate = MagicMock(
            return_value=RiskOutput(
                decision=RiskDecision.APPROVED,
                reason="HOLD approved",
                position_size=None,
                stop_loss_price=None,
                take_profit_price=None,
                atr_at_evaluation=Decimal(15000),
                max_capital_pct=Decimal("0.05"),
            )
        )
        executor = _mock_executor()
        executor.execute = AsyncMock(return_value=None)

        manager = ExecutionManager(
            inference_engine=inf_engine,
            risk_engine=risk_engine,
            executor=executor,
            portfolio_provider=_mock_portfolio_provider(),
        )

        result = await manager.run(inference_input=_make_inference_input())

        executor.execute.assert_awaited_once()
        assert result is None  # HOLD → no execution result

    async def test_rejected_signal_does_not_execute(self):
        risk_engine = _mock_risk_engine()
        risk_engine.evaluate = MagicMock(
            return_value=RiskOutput(
                decision=RiskDecision.REJECTED,
                reason="Drawdown exceeded",
                position_size=None,
                stop_loss_price=None,
                take_profit_price=None,
                atr_at_evaluation=Decimal(15000),
                max_capital_pct=None,
            )
        )
        executor = _mock_executor()
        executor.execute = AsyncMock(return_value=None)

        manager = ExecutionManager(
            inference_engine=_mock_inference_engine(),
            risk_engine=risk_engine,
            executor=executor,
            portfolio_provider=_mock_portfolio_provider(),
        )

        result = await manager.run(inference_input=_make_inference_input())

        executor.execute.assert_awaited_once()
        assert result is None


class TestPipelineCallsInOrder:
    async def test_inference_called_first(self):
        inf_engine = _mock_inference_engine()
        manager = ExecutionManager(
            inference_engine=inf_engine,
            risk_engine=_mock_risk_engine(),
            executor=_mock_executor(),
            portfolio_provider=_mock_portfolio_provider(),
        )

        await manager.run(inference_input=_make_inference_input())

        inf_engine.infer.assert_called_once()

    async def test_risk_engine_called_with_correct_input(self):
        risk_engine = _mock_risk_engine()
        manager = ExecutionManager(
            inference_engine=_mock_inference_engine(),
            risk_engine=risk_engine,
            executor=_mock_executor(),
            portfolio_provider=_mock_portfolio_provider(),
        )

        await manager.run(inference_input=_make_inference_input())

        risk_engine.evaluate.assert_called_once()
        risk_input: RiskInput = risk_engine.evaluate.call_args.args[0]
        assert isinstance(risk_input, RiskInput)
        assert risk_input.signal_action == SignalAction.BUY
        assert risk_input.confidence == Decimal("0.85")
        assert risk_input.current_price == Decimal(500000)
        assert risk_input.atr == Decimal(15000)

    async def test_executor_called_with_risk_output(self):
        executor = _mock_executor()
        manager = ExecutionManager(
            inference_engine=_mock_inference_engine(),
            risk_engine=_mock_risk_engine(),
            executor=executor,
            portfolio_provider=_mock_portfolio_provider(),
        )

        await manager.run(inference_input=_make_inference_input())

        executor.execute.assert_awaited_once()
        kwargs = executor.execute.call_args.kwargs
        assert kwargs["symbol"] == "btc_mxn"
        assert kwargs["signal_action"] == SignalAction.BUY
        assert kwargs["venue"] == MarketVenue.BITSO