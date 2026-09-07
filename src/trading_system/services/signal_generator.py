"""
Generador de señales basado en reglas técnicas.

Dos reglas de entrada BUY:

  - Regla clásica (RSI + MACD):
      BUY cuando RSI < oversold_threshold Y MACD diff > 0
      (cruce alcista en zona de sobreventa)

  - Regla de crecimiento (EMA + volumen):
      BUY cuando EMA rápida > EMA lenta Y MACD diff > 0 Y
      volume_ratio > volume_threshold (tendencia alcista confirmada
      por volumen inusualmente alto)

  - SELL cuando RSI > overbought_threshold Y MACD diff < 0
  - HOLD en cualquier otro caso

La confianza se calcula según qué tan extremo es el disparador
(RSI para la regla clásica, volume_ratio para la de crecimiento):
  - extremo → confianza ~0.95
  - threshold → confianza ~0.60
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
    """Genera señales BUY/SELL/HOLD basadas en RSI, MACD, EMA y volumen."""

    def __init__(
        self,
        rsi_oversold: float = 30.0,
        rsi_overbought: float = 70.0,
        rsi_extreme_low: float = 20.0,
        rsi_extreme_high: float = 80.0,
        min_confidence: float = 0.60,
        max_confidence: float = 0.95,
        volume_threshold: float = 1.5,
        volume_extreme: float = 4.0,
    ) -> None:
        self._oversold = rsi_oversold
        self._overbought = rsi_overbought
        self._extreme_low = rsi_extreme_low
        self._extreme_high = rsi_extreme_high
        self._min_conf = min_confidence
        self._max_conf = max_confidence
        self._vol_threshold = volume_threshold
        self._vol_extreme = volume_extreme

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

        growth = self._check_ema_volume(features, rsi, macd_diff)
        if growth is not None:
            return growth

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

    def _check_ema_volume(
        self,
        features: dict[str, float],
        rsi: float,
        macd_diff: float,
    ) -> RuleSignal | None:
        """Regla de crecimiento: EMA rápida > EMA lenta + MACD alcista + volumen alto."""
        ema_fast = features.get("ema_9")
        ema_slow = features.get("ema_21")
        volume_ratio = features.get("volume_ratio_20")

        if ema_fast is None or ema_slow is None or volume_ratio is None:
            return None

        if ema_fast <= ema_slow or macd_diff <= 0:
            return None

        if volume_ratio <= self._vol_threshold:
            return None

        ratio = (volume_ratio - self._vol_threshold) / (self._vol_extreme - self._vol_threshold)
        ratio = min(max(ratio, 0.0), 1.0)
        confidence = self._min_conf + ratio * (self._max_conf - self._min_conf)

        return RuleSignal(
            action=SignalAction.BUY,
            confidence=Decimal(str(round(confidence, 4))),
            rsi=rsi,
            macd_diff=macd_diff,
            reason=(
                f"Crecimiento: EMA9>EMA21 ({ema_fast:,.0f} vs {ema_slow:,.0f}) "
                f"+ MACD alcista + volumen {volume_ratio:.1f}x"
            ),
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
