"""TDD — Tests for TechnicalModel ABC and XGBoostTechnicalModel (RED phase).

``TechnicalModel`` es el contrato: dado un dict de features, devuelve
probabilidades por clase [buy, hold, sell].

``XGBoostTechnicalModel`` delega a un booster de XGBoost, que se inyecta
por constructor (DIP) para tests.
"""
from decimal import Decimal
from unittest.mock import MagicMock

import pytest

from trading_system.core.enums import SignalAction
from trading_system.inference.models.base import TechnicalModel, TechnicalResult
from trading_system.inference.models.xgboost_model import XGBoostTechnicalModel


class TestTechnicalModelABC:
    def test_cannot_instantiate_directly(self):
        with pytest.raises(TypeError):
            TechnicalModel()  # type: ignore[abstract]


class TestXGBoostTechnicalModel:
    def test_predict_returns_technical_result(self):
        mock_booster = MagicMock()
        mock_booster.predict_proba = MagicMock(return_value=[[0.7, 0.2, 0.1]])
        model = XGBoostTechnicalModel(
            booster=mock_booster,
            feature_names=["rsi_14", "atr_14", "macd"],
        )
        result = model.predict({"rsi_14": 65.0, "atr_14": 15000.0, "macd": 120.5})

        assert isinstance(result, TechnicalResult)
        assert result.action == SignalAction.BUY
        assert result.confidence == Decimal("0.7")

    def test_predict_sell_when_sell_prob_highest(self):
        mock_booster = MagicMock()
        mock_booster.predict_proba = MagicMock(return_value=[[0.1, 0.2, 0.7]])
        model = XGBoostTechnicalModel(
            booster=mock_booster,
            feature_names=["rsi_14"],
        )
        result = model.predict({"rsi_14": 20.0})
        assert result.action == SignalAction.SELL
        assert result.confidence == Decimal("0.7")

    def test_predict_hold_when_hold_prob_highest(self):
        mock_booster = MagicMock()
        mock_booster.predict_proba = MagicMock(return_value=[[0.2, 0.6, 0.2]])
        model = XGBoostTechnicalModel(
            booster=mock_booster,
            feature_names=["rsi_14"],
        )
        result = model.predict({"rsi_14": 50.0})
        assert result.action == SignalAction.HOLD
        assert result.confidence == Decimal("0.6")

    def test_predict_missing_feature_raises(self):
        mock_booster = MagicMock()
        model = XGBoostTechnicalModel(
            booster=mock_booster,
            feature_names=["rsi_14", "atr_14"],
        )
        with pytest.raises(KeyError, match="atr_14"):
            model.predict({"rsi_14": 50.0})

    def test_model_name_property(self):
        model = XGBoostTechnicalModel(booster=MagicMock(), feature_names=[])
        assert "xgboost" in model.model_name.lower()