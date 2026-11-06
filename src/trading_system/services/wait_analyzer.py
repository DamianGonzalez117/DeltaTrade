"""
Análisis estadístico de "¿qué pasaría si espero?" para decisiones de venta.

Dado el OHLCV reciente del libro, busca momentos históricos con RSI similar
al actual y mide qué pasó en las siguientes velas del **horizonte
adaptativo**: se mira más lejos mientras más historial haya disponible
(``len(ohlcv) // history_divisor`` velas, acotado a
``[horizon_candles, max_horizon_candles]``), de modo que con timeframe 1m y
poco historial se analizan ~6 velas y con historial abundante hasta 48.

  - ``rise_share``: fracción de casos donde el precio terminó más alto.
  - ``fall_share``: fracción donde terminó más bajo.
  - ``avg_gain_pct`` / ``avg_loss_pct``: retorno final promedio de cada lado.
  - ``best_case_pct`` / ``worst_case_pct``: máximo alcance alcista y máxima
    caída observados dentro del horizonte.
  - ``trend``: EMA9 vs EMA21 actual (alcista/bajista/neutral).
  - ``recommendation``: "esperar" | "vender" | "neutro" según valor
    esperado del retorno adicional.

DIP: recibe el DataFrame OHLCV (producido por el monitor/ingestor).
SRP: solo estadística. No conversa ni ejecuta.
"""
from __future__ import annotations

from dataclasses import dataclass

import pandas as pd
from ta.momentum import RSIIndicator
from ta.trend import EMAIndicator


@dataclass(frozen=True, slots=True)
class WaitStats:
    """Estadística de escenarios de espera para un libro."""

    n_cases: int
    horizon_candles: int  # velas efectivas miradas hacia adelante
    horizon_minutes: int
    rise_share: float  # 0..1
    fall_share: float  # 0..1
    avg_gain_pct: float  # retorno final promedio de los casos que subieron
    avg_loss_pct: float  # retorno final promedio (negativo) de los que bajaron
    best_case_pct: float  # máximo alcance alcista observado en el horizonte
    worst_case_pct: float  # máxima caída observada en el horizonte
    trend: str  # "alcista" | "bajista" | "neutral"
    recommendation: str  # "esperar" | "vender" | "neutro"


class WaitAnalyzer:
    """Calcula ``WaitStats`` a partir de OHLCV y el RSI actual."""

    def __init__(
        self,
        *,
        horizon_candles: int = 6,
        max_horizon_candles: int = 48,
        history_divisor: int = 8,
        min_cases: int = 10,
        rsi_band: float = 7.5,
        rsi_period: int = 14,
    ) -> None:
        if horizon_candles <= 0:
            raise ValueError("horizon_candles debe ser positivo")
        if max_horizon_candles < horizon_candles:
            raise ValueError(
                "max_horizon_candles no puede ser menor que horizon_candles"
            )
        if history_divisor <= 0:
            raise ValueError("history_divisor debe ser positivo")
        self._min_horizon = horizon_candles
        self._max_horizon = max_horizon_candles
        self._divisor = history_divisor
        self._min_cases = min_cases
        self._rsi_band = rsi_band
        self._rsi_period = rsi_period

    def _effective_horizon(self, n_candles: int) -> int:
        """Horizonte adaptativo: crece con el historial, acotado al rango."""
        target = n_candles // self._divisor
        return max(self._min_horizon, min(self._max_horizon, target))

    def analyze(
        self,
        ohlcv: pd.DataFrame | None,
        *,
        current_rsi: float | None,
        timeframe_minutes: int,
    ) -> WaitStats | None:
        """Retorna las estadísticas, o None si no hay datos suficientes.

        Si ``current_rsi`` es None se usa el último RSI de la serie. La banda
        de similitud se ensancha progresivamente hasta juntar ``min_cases``.
        """
        if ohlcv is None or ohlcv.empty:
            return None

        if len(ohlcv) < self._rsi_period + self._min_horizon + 10:
            return None

        closes = ohlcv["close"]
        rsi_series = RSIIndicator(close=closes, window=self._rsi_period).rsi()
        valid_rsi = rsi_series.dropna()
        if valid_rsi.empty:
            return None

        rsi_now = float(valid_rsi.iloc[-1]) if current_rsi is None else float(current_rsi)
        horizon = self._effective_horizon(len(ohlcv))

        cases: list[tuple[float, float, float]] = []
        for band in (self._rsi_band, self._rsi_band * 2, self._rsi_band * 4):
            cases = self._collect_cases(ohlcv, rsi_series, rsi_now, band, horizon)
            if len(cases) >= self._min_cases:
                break
        if not cases:
            return None

        finals = [final for final, _, _ in cases]
        ups = [up for _, up, _ in cases]
        downs = [down for _, _, down in cases]

        rises = [r for r in finals if r > 0]
        falls = [r for r in finals if r <= 0]
        n = len(finals)
        rise_share = len(rises) / n
        fall_share = 1.0 - rise_share
        avg_gain_pct = (sum(rises) / len(rises) * 100) if rises else 0.0
        avg_loss_pct = (sum(falls) / len(falls) * 100) if falls else 0.0

        return WaitStats(
            n_cases=n,
            horizon_candles=horizon,
            horizon_minutes=horizon * timeframe_minutes,
            rise_share=rise_share,
            fall_share=fall_share,
            avg_gain_pct=avg_gain_pct,
            avg_loss_pct=avg_loss_pct,
            best_case_pct=max(ups) * 100,
            worst_case_pct=min(downs) * 100,
            trend=_trend(closes),
            recommendation=_recommend(rise_share, fall_share, avg_gain_pct, avg_loss_pct),
        )

    def _collect_cases(
        self,
        ohlcv: pd.DataFrame,
        rsi_series: pd.Series,
        rsi_now: float,
        band: float,
        horizon: int,
    ) -> list[tuple[float, float, float]]:
        """Casos históricos con |RSI − RSI actual| <= band.

        Cada caso: (retorno final a ``horizon`` velas, máximo alcance con
        highs, máxima caída con lows).
        """
        closes = ohlcv["close"].reset_index(drop=True)
        highs = ohlcv["high"].reset_index(drop=True)
        lows = ohlcv["low"].reset_index(drop=True)
        rsi = rsi_series.reset_index(drop=True)

        cases: list[tuple[float, float, float]] = []
        last_start = len(closes) - horizon - 1
        for i in range(self._rsi_period, last_start):
            rsi_i = rsi.iloc[i]
            if pd.isna(rsi_i) or abs(float(rsi_i) - rsi_now) > band:
                continue
            entry = float(closes.iloc[i])
            if entry <= 0:
                continue
            window = range(i + 1, i + 1 + horizon)
            exit_price = float(closes.iloc[i + horizon])
            max_up = max(float(highs.iloc[j]) for j in window) / entry - 1.0
            max_down = min(float(lows.iloc[j]) for j in window) / entry - 1.0
            cases.append((exit_price / entry - 1.0, max_up, max_down))
        return cases


def _trend(closes: pd.Series) -> str:
    ema_fast = EMAIndicator(close=closes, window=9).ema_indicator().iloc[-1]
    ema_slow = EMAIndicator(close=closes, window=21).ema_indicator().iloc[-1]
    if pd.notna(ema_fast) and pd.notna(ema_slow):
        if ema_fast > ema_slow * 1.001:
            return "alcista"
        if ema_fast < ema_slow * 0.999:
            return "bajista"
    return "neutral"


def _recommend(
    rise_share: float,
    fall_share: float,
    avg_gain_pct: float,
    avg_loss_pct: float,
) -> str:
    """Recomendación simple por valor esperado del retorno adicional."""
    expected = rise_share * avg_gain_pct + fall_share * avg_loss_pct
    if expected > 0.3 and rise_share > 0.5:
        return "esperar"
    if expected < -0.3 or fall_share >= 0.6:
        return "vender"
    return "neutro"
