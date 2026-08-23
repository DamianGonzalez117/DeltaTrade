"""
Generador de señales basado en reglas técnicas (RSI + MACD).

Para el prototipo no se requiere un modelo XGBoost entrenado. Este
generador usa reglas clásicas de trading:

  - BUY  cuando RSI < oversold_threshold  Y  MACD diff > 0  (cruce alcista en zona de sobreventa)
  - SELL cuando RSI > overbought_threshold Y  MACD diff < 0  (cruce bajista en zona de sobrecompra)
  - HOLD en cualquier otro caso

La confianza se calcula según qué tan extremo es el RSI:
  - RSI en 20/80 → confianza ~0.95
  - RSI en 30/70 → confianza ~0.75
  - Interpolación lineal entre el threshold y el extremo.

DIP: recibe features como dict plano (mismo formato que TechnicalFeaturesPipeline.extract_latest).
SRP: solo genera la señal. No decide tamaño, no ejecuta, no notifica.
"""
from __future__ import annotations

from dataclasses import dataclass
from decimal import Decimal

from trading_system.core.enums import SignalAction


@dataclass(frozen=True, slots=True)
class RuleSignal:
    """Señal generada por reglas técnicas."""

    action: SignalAction
    confidence: Decimal
    rsi: float
    macd_diff: float
    reason: str


class SignalGenerator:
    """Genera señales BUY/SELL/HOLD basadas en RSI y MACD."""

    def __init__(
        self,
        rsi_oversold: float = 30.0,
        rsi_overbought: float = 70.0,
        rsi_extreme_low: float = 20.0,
        rsi_extreme_high: float = 80.0,
        min_confidence: float = 0.60,
        max_confidence: float = 0.95,
    ) -> None:
        self._oversold = rsi_oversold
        self._overbought = rsi_overbought
        self._extreme_low = rsi_extreme_low
        self._extreme_high = rsi_extreme_high
        self._min_conf = min_confidence
        self._max_conf = max_confidence

    def generate(self, features: dict[str, float]) -> RuleSignal:
        """Genera una señal a partir de features técnicas."""
        rsi = features.get("rsi_14", 50.0)
        macd_diff = features.get("macd_diff", 0.0)

        if rsi < self._oversold and macd_diff > 0:
            confidence = self._calc_confidence_buy(rsi)
            return RuleSignal(
                action=SignalAction.BUY,
                confidence=Decimal(str(round(confidence, 4))),
                rsi=rsi,
                macd_diff=macd_diff,
                reason=f"RSI sobreventa ({rsi:.1f}) + MACD cruce alcista",
            )

        if rsi > self._overbought and macd_diff < 0:
            confidence = self._calc_confidence_sell(rsi)
            return RuleSignal(
                action=SignalAction.SELL,
                confidence=Decimal(str(round(confidence, 4))),
                rsi=rsi,
                macd_diff=macd_diff,
                reason=f"RSI sobrecompra ({rsi:.1f}) + MACD cruce bajista",
            )

        return RuleSignal(
            action=SignalAction.HOLD,
            confidence=Decimal("0.50"),
            rsi=rsi,
            macd_diff=macd_diff,
            reason=f"RSI neutral ({rsi:.1f}), sin cruce MACD decisivo",
        )

    def _calc_confidence_buy(self, rsi: float) -> float:
        """Mayor confianza cuanto menor sea el RSI (más sobreventa)."""
        if rsi <= self._extreme_low:
            return self._max_conf
        ratio = (self._oversold - rsi) / (self._oversold - self._extreme_low)
        return self._min_conf + ratio * (self._max_conf - self._min_conf)

    def _calc_confidence_sell(self, rsi: float) -> float:
        """Mayor confianza cuanto mayor sea el RSI (más sobrecompra)."""
        if rsi >= self._extreme_high:
            return self._max_conf
        ratio = (rsi - self._overbought) / (self._extreme_high - self._overbought)
        return self._min_conf + ratio * (self._max_conf - self._min_conf)
