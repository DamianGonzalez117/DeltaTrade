"""
Motor de backtesting para validación de estrategias.

Recorre datos OHLCV históricos candle por candle:
  1. Calcula features técnicas hasta esa vela.
  2. Ejecuta el modelo de inferencia (inyectado).
  3. Evalúa con el motor de riesgo (inyectado).
  4. Simula ejecución (sin órdenes reales).
  5. Calcula métricas de desempeño.

DIP: ``TechnicalFeaturesPipeline``, ``predict_fn``, ``RiskEngine`` se inyectan.
SRP: solo simula. No toca la red ni la DB.
"""
from __future__ import annotations

from collections.abc import Callable
from dataclasses import dataclass, field
from decimal import Decimal

import numpy as np
import pandas as pd
import structlog

from trading_system.core.enums import MarketVenue, SignalAction
from trading_system.features.pipeline import TechnicalFeaturesPipeline
from trading_system.risk.engine import RiskEngine
from trading_system.risk.types import RiskInput, RiskOutput

logger = structlog.get_logger(__name__)


@dataclass(frozen=True, slots=True)
class BacktestConfig:
    """Configuración del backtest."""

    initial_capital: Decimal = Decimal(10000)
    fee_pct: Decimal = Decimal("0.001")  # 0.1% por trade
    slippage_pct: Decimal = Decimal("0.0005")  # 0.05% slippage
    max_capital_pct: Decimal = Decimal("0.05")
    max_drawdown_pct: Decimal = Decimal("0.20")


@dataclass(frozen=True, slots=True)
class BacktestResult:
    """Resultado del backtest con métricas de desempeño."""

    total_return_pct: float
    max_drawdown_pct: float
    sharpe_ratio: float
    num_trades: int
    num_wins: int
    num_losses: int
    win_rate: float
    final_equity: Decimal
    equity_curve: list[float] = field(default_factory=list)
    trades: list[dict] = field(default_factory=list)


@dataclass
class _PortfolioState:
    """Estado mutable del portafolio durante el backtest."""

    cash: Decimal
    position: Decimal = Decimal(0)
    entry_price: Decimal = Decimal(0)
    peak_equity: Decimal = Decimal(0)
    trades: list[dict] = field(default_factory=list)
    equity_curve: list[float] = field(default_factory=list)

    @property
    def equity(self) -> Decimal:
        return self.cash + self.position

    def update_peak(self, price: Decimal) -> None:
        current = self.cash + self.position * price
        self.peak_equity = max(self.peak_equity, current)
        self.equity_curve.append(float(current))


class BacktestEngine:
    """Simula el pipeline completo sobre datos históricos."""

    def __init__(
        self,
        *,
        features_pipeline: TechnicalFeaturesPipeline,
        predict_fn: Callable[[dict[str, float]], tuple[SignalAction, Decimal]],
        risk_engine: RiskEngine,
        config: BacktestConfig | None = None,
    ) -> None:
        self._pipeline = features_pipeline
        self._predict_fn = predict_fn
        self._risk = risk_engine
        self._config = config or BacktestConfig()

    def run(self, ohlcv: pd.DataFrame) -> BacktestResult:
        """Ejecuta el backtest sobre el DataFrame OHLCV."""
        if len(ohlcv) < 30:
            raise ValueError(f"Insuficientes velas para backtest: {len(ohlcv)} (mínimo 30)")

        state = _PortfolioState(cash=self._config.initial_capital)
        state.peak_equity = self._config.initial_capital

        features_df = self._pipeline.calculate_all(ohlcv)

        for i in range(len(ohlcv)):
            row = ohlcv.iloc[i]
            price = Decimal(str(row["close"]))

            if pd.isna(features_df.iloc[i]).any():
                state.update_peak(price)
                continue

            features = {
                col: float(features_df.iloc[i][col])
                for col in features_df.columns
                if pd.notna(features_df.iloc[i][col])
            }

            action, confidence = self._predict_fn(features)

            risk_input = RiskInput(
                signal_action=action,
                confidence=confidence,
                asset_symbol="backtest",
                asset_venue=MarketVenue.BITSO,
                current_price=price,
                atr=Decimal(str(features.get("atr_14", 0))) if "atr_14" in features else None,
                portfolio_equity=state.equity,
                current_exposure=state.position * price,
                max_capital_pct=self._config.max_capital_pct,
                max_drawdown_pct=self._config.max_drawdown_pct,
                current_drawdown=self._calc_drawdown(state, price),
                is_market_open=True,
            )

            risk_output = self._risk.evaluate(risk_input)

            if risk_output.decision.value in ("approved", "adjusted"):
                self._execute_sim(state, action, risk_output, price)

            state.update_peak(price)

        return self._compute_result(state, ohlcv)

    def _calc_drawdown(self, state: _PortfolioState, price: Decimal) -> Decimal:
        if state.peak_equity == 0:
            return Decimal(0)
        current = state.cash + state.position * price
        dd = (state.peak_equity - current) / state.peak_equity
        return dd if dd > 0 else Decimal(0)

    def _execute_sim(
        self,
        state: _PortfolioState,
        action: SignalAction,
        risk_output: RiskOutput,
        price: Decimal,
    ) -> None:
        if risk_output.position_size is None or risk_output.position_size == 0:
            return

        if action == SignalAction.BUY and state.position == 0:
            exec_price = price * (Decimal(1) + self._config.slippage_pct)
            qty = risk_output.position_size
            cost = qty * exec_price
            fee = cost * self._config.fee_pct
            if cost + fee > state.cash:
                return
            state.cash -= cost + fee
            state.position = qty
            state.entry_price = exec_price
            state.trades.append({
                "side": "buy",
                "price": float(exec_price),
                "qty": float(qty),
                "fee": float(fee),
                "ts": len(state.equity_curve),
            })

        elif action == SignalAction.SELL and state.position > 0:
            exec_price = price * (Decimal(1) - self._config.slippage_pct)
            qty = state.position
            revenue = qty * exec_price
            fee = revenue * self._config.fee_pct
            pnl = (exec_price - state.entry_price) * qty - fee
            state.cash += revenue - fee
            state.trades.append({
                "side": "sell",
                "price": float(exec_price),
                "qty": float(qty),
                "fee": float(fee),
                "pnl": float(pnl),
                "ts": len(state.equity_curve),
            })
            state.position = Decimal(0)
            state.entry_price = Decimal(0)

    def _compute_result(
        self,
        state: _PortfolioState,
        ohlcv: pd.DataFrame,
    ) -> BacktestResult:
        final_price = Decimal(str(ohlcv.iloc[-1]["close"]))
        final_equity = state.cash + state.position * final_price

        total_return = float(
            (final_equity - self._config.initial_capital) / self._config.initial_capital * 100
        )

        equity_arr = np.array(state.equity_curve) if state.equity_curve else np.array([0.0])
        peak = np.maximum.accumulate(equity_arr)
        drawdowns = (peak - equity_arr) / peak * 100
        max_dd = float(np.max(drawdowns)) if len(drawdowns) > 0 else 0.0

        if len(equity_arr) > 1:
            returns = np.diff(equity_arr) / equity_arr[:-1]
            std = np.std(returns)
            sharpe = float(np.mean(returns) / std * np.sqrt(252)) if std > 0 else 0.0
        else:
            sharpe = 0.0

        sells = [t for t in state.trades if t["side"] == "sell"]
        wins = sum(1 for t in sells if t.get("pnl", 0) > 0)
        losses = sum(1 for t in sells if t.get("pnl", 0) <= 0)
        total_sells = len(sells)
        win_rate = wins / total_sells if total_sells > 0 else 0.0

        return BacktestResult(
            total_return_pct=total_return,
            max_drawdown_pct=max_dd,
            sharpe_ratio=sharpe,
            num_trades=len(state.trades),
            num_wins=wins,
            num_losses=losses,
            win_rate=win_rate,
            final_equity=final_equity,
            equity_curve=state.equity_curve,
            trades=state.trades,
        )