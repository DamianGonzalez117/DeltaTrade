"""
Manager de ejecución: orquesta el pipeline completo del sistema.

Flujo:
  1. ``HybridInferenceEngine.infer(InferenceInput)`` → ``InferenceOutput``
  2. Obtiene estado del portafolio via ``PortfolioProvider``
  3. Construye ``RiskInput`` combinando inferencia + portafolio
  4. ``RiskEngine.evaluate(RiskInput)`` → ``RiskOutput``
  5. ``OrderExecutor.execute(RiskOutput, ...)`` → ``ExecutionResult | None``

DIP: todas las dependencias se inyectan.
SRP: solo orquesta — no decide, no ejecuta, no infiere.
"""
from __future__ import annotations

from abc import ABC, abstractmethod

import structlog

from trading_system.connectors.base import ExecutionResult
from trading_system.execution.executor import OrderExecutor
from trading_system.inference.engine import HybridInferenceEngine
from trading_system.inference.types import InferenceInput
from trading_system.risk.engine import RiskEngine
from trading_system.risk.types import RiskInput

logger = structlog.get_logger(__name__)


class PortfolioProvider(ABC):
    """Proveedor abstracto del estado del portafolio (para inyectar)."""

    @abstractmethod
    def get_portfolio_state(self, asset_symbol: str, asset_venue: str) -> dict:
        """Retorna equity, exposure, drawdown, price, ATR, etc."""
        ...


class ExecutionManager:
    """Orquesta inferencia → riesgo → ejecución."""

    def __init__(
        self,
        *,
        inference_engine: HybridInferenceEngine,
        risk_engine: RiskEngine,
        executor: OrderExecutor,
        portfolio_provider: PortfolioProvider,
    ) -> None:
        self._inference = inference_engine
        self._risk = risk_engine
        self._executor = executor
        self._portfolio = portfolio_provider

    async def run(
        self,
        *,
        inference_input: InferenceInput,
    ) -> ExecutionResult | None:
        inference_output = self._inference.infer(inference_input)
        logger.info(
            "manager.inference_done",
            symbol=inference_input.asset_symbol,
            action=inference_output.action.value,
            confidence=str(inference_output.confidence),
        )

        portfolio_state = self._portfolio.get_portfolio_state(
            asset_symbol=inference_input.asset_symbol,
            asset_venue=inference_input.asset_venue.value,
        )

        risk_input = RiskInput(
            signal_action=inference_output.action,
            confidence=inference_output.confidence,
            asset_symbol=inference_input.asset_symbol,
            asset_venue=inference_input.asset_venue,
            current_price=portfolio_state["current_price"],
            atr=portfolio_state.get("atr"),
            portfolio_equity=portfolio_state["equity"],
            current_exposure=portfolio_state["current_exposure"],
            max_capital_pct=portfolio_state["max_capital_pct"],
            max_drawdown_pct=portfolio_state["max_drawdown_pct"],
            current_drawdown=portfolio_state["current_drawdown"],
            is_market_open=portfolio_state.get("is_market_open"),
        )

        risk_output = self._risk.evaluate(risk_input)
        logger.info(
            "manager.risk_done",
            symbol=inference_input.asset_symbol,
            decision=risk_output.decision.value,
        )

        result = await self._executor.execute(
            risk_output=risk_output,
            symbol=inference_input.asset_symbol,
            signal_action=inference_output.action,
            venue=inference_input.asset_venue,
        )

        logger.info(
            "manager.execution_done",
            symbol=inference_input.asset_symbol,
            executed=result is not None,
        )

        return result