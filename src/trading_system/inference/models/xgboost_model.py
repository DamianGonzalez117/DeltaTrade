"""Modelo técnico basado en XGBoost.

DIP: el booster de XGBoost se inyecta por constructor para tests.
SRP: solo mapea features → probabilidades → SignalAction.
"""
from decimal import Decimal

from trading_system.core.enums import SignalAction
from trading_system.inference.models.base import TechnicalModel, TechnicalResult

_ACTION_ORDER = [SignalAction.BUY, SignalAction.HOLD, SignalAction.SELL]


class XGBoostTechnicalModel(TechnicalModel):
    """Modelo XGBoost para predicción de señales buy/hold/sell."""

    def __init__(
        self,
        booster,
        feature_names: list[str],
    ) -> None:
        self._booster = booster
        self._feature_names = feature_names

    @property
    def model_name(self) -> str:
        return "xgboost_v1"

    def predict(self, features: dict[str, float]) -> TechnicalResult:
        for name in self._feature_names:
            if name not in features:
                raise KeyError(name)

        ordered_features = [features[name] for name in self._feature_names]
        probas = self._booster.predict_proba([ordered_features])[0]
        best_idx = max(range(len(probas)), key=lambda i: probas[i])
        action = _ACTION_ORDER[best_idx]
        confidence = Decimal(str(probas[best_idx]))
        return TechnicalResult(action=action, confidence=confidence)