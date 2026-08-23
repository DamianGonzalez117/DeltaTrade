"""
Pipeline de entrenamiento del modelo XGBoost.

Pasos:
  1. Cargar OHLCV desde la DB (o aceptar un DataFrame inyectado).
  2. Calcular features técnicas con ``TechnicalFeaturesPipeline``.
  3. Etiquetar con *forward returns* → BUY / HOLD / SELL.
  4. Entrenar XGBoost (multiclase, 3 clases).
  5. Persistir el booster a disco (formato JSON de XGBoost).

DIP: el ``TechnicalFeaturesPipeline`` se inyecta. ``xgboost.XGBClassifier``
se inyecta como factory para tests.
SRP: solo entrena y persiste. No decide ni ejecuta órdenes.
"""
from __future__ import annotations

from collections.abc import Callable
from pathlib import Path
from typing import Protocol

import numpy as np
import pandas as pd
import structlog
from sqlalchemy import select
from sqlalchemy.orm import Session

from trading_system.core.enums import SignalAction, TimeFrame
from trading_system.db.models import Asset, MarketDataOHLCV
from trading_system.db.session import get_session
from trading_system.features.pipeline import TechnicalFeaturesPipeline

logger = structlog.get_logger(__name__)

_ACTION_TO_LABEL: dict[SignalAction, int] = {
    SignalAction.BUY: 0,
    SignalAction.HOLD: 1,
    SignalAction.SELL: 2,
}
_LABEL_TO_ACTION: dict[int, SignalAction] = {v: k for k, v in _ACTION_TO_LABEL.items()}


class XGBClassifierLike(Protocol):
    """Contrato mínimo que cumple ``xgboost.XGBClassifier``."""

    def fit(self, X: np.ndarray, y: np.ndarray) -> None: ...
    def predict_proba(self, X: np.ndarray) -> np.ndarray: ...
    def save_model(self, path: str) -> None: ...
    def get_booster(self): ...


class TrainingPipeline:
    """Entrena un modelo XGBoost sobre features técnicas + forward-return labels."""

    def __init__(
        self,
        features_pipeline: TechnicalFeaturesPipeline,
        classifier_factory: Callable[..., XGBClassifierLike],
        forward_window: int = 5,
        buy_threshold: float = 0.01,
        sell_threshold: float = -0.01,
    ) -> None:
        self._pipeline = features_pipeline
        self._classifier_factory = classifier_factory
        self._forward_window = forward_window
        self._buy_threshold = buy_threshold
        self._sell_threshold = sell_threshold
        self._feature_names: list[str] = []
        self._classifier: XGBClassifierLike | None = None

    @property
    def feature_names(self) -> list[str]:
        return list(self._feature_names)

    @staticmethod
    def load_ohlcv_from_db(
        session: Session,
        asset_id: int,
        timeframe: TimeFrame,
    ) -> pd.DataFrame:
        """Carga velas OHLCV desde la DB y las devuelve como DataFrame pandas."""
        rows = session.execute(
            select(MarketDataOHLCV)
            .where(
                MarketDataOHLCV.asset_id == asset_id,
                MarketDataOHLCV.timeframe == str(timeframe.value),
            )
            .order_by(MarketDataOHLCV.ts.asc())
        ).scalars().all()

        if not rows:
            return pd.DataFrame(
                columns=["ts", "open", "high", "low", "close", "volume"]
            )

        return pd.DataFrame([
            {
                "ts": r.ts,
                "open": float(r.open),
                "high": float(r.high),
                "low": float(r.low),
                "close": float(r.close),
                "volume": float(r.volume),
            }
            for r in rows
        ])

    @staticmethod
    def load_ohlcv_from_df(df: pd.DataFrame) -> pd.DataFrame:
        """Normaliza un DataFrame OHLCV externo al formato interno."""
        required = {"open", "high", "low", "close", "volume"}
        missing = required - set(df.columns)
        if missing:
            raise ValueError(f"Columnas faltantes: {missing}")
        if "ts" not in df.columns:
            df = df.copy()
            df["ts"] = df.index if df.index.name == "ts" else range(len(df))
        return df[list(required | {"ts"})]

    def label_forward_returns(self, df: pd.DataFrame) -> pd.DataFrame:
        """Añade columna 'label' (0=BUY, 1=HOLD, 2=SELL) basada en forward returns."""
        df = df.copy()
        df["forward_return"] = (
            df["close"].shift(-self._forward_window) / df["close"] - 1
        )
        df["label"] = np.where(
            df["forward_return"] > self._buy_threshold,
            _ACTION_TO_LABEL[SignalAction.BUY],
            np.where(
                df["forward_return"] < self._sell_threshold,
                _ACTION_TO_LABEL[SignalAction.SELL],
                _ACTION_TO_LABEL[SignalAction.HOLD],
            ),
        )
        df = df.dropna(subset=["forward_return"])
        return df

    def prepare_training_data(
        self,
        ohlcv: pd.DataFrame,
    ) -> tuple[np.ndarray, np.ndarray]:
        """Combina features + labels y devuelve (X, y) listos para entrenar."""
        if ohlcv.empty:
            raise ValueError("No hay filas válidas después de dropna")

        features = self._pipeline.calculate_all(ohlcv)
        labeled = self.label_forward_returns(ohlcv)

        combined = features.loc[labeled.index].copy()
        combined["label"] = labeled["label"].values

        combined = combined.dropna()

        if combined.empty:
            raise ValueError("No hay filas válidas después de dropna")

        self._feature_names = [
            c for c in combined.columns if c != "label"
        ]

        X = combined[self._feature_names].to_numpy(dtype=np.float64)
        y = combined["label"].to_numpy(dtype=np.int64)
        return X, y

    def train(self, ohlcv: pd.DataFrame) -> XGBClassifierLike:
        """Entrena el clasificador sobre los datos OHLCV proporcionados."""
        X, y = self.prepare_training_data(ohlcv)

        if len(X) < 10:
            raise ValueError(
                f"Insuficientes muestras para entrenar: {len(X)} (mínimo 10)"
            )

        classifier = self._classifier_factory(
            n_estimators=100,
            max_depth=4,
            learning_rate=0.1,
            objective="multi:softprob",
            num_class=3,
            eval_metric="mlogloss",
            random_state=42,
        )
        classifier.fit(X, y)
        self._classifier = classifier

        unique, counts = np.unique(y, return_counts=True)
        logger.info(
            "train.complete",
            samples=len(X),
            classes=dict(zip(unique.tolist(), counts.tolist())),
            features=len(self._feature_names),
        )
        return classifier

    def save_model(self, path: str | Path) -> None:
        """Persiste el booster y los feature_names a disco."""
        if self._classifier is None:
            raise RuntimeError("Modelo no entrenado. Llama train() primero.")

        path = Path(path)
        path.parent.mkdir(parents=True, exist_ok=True)
        self._classifier.save_model(str(path))

        meta_path = path.with_suffix(".features.json")
        import json
        meta_path.write_text(
            json.dumps(
                {
                    "feature_names": self._feature_names,
                    "label_mapping": {
                        str(k.value): v
                        for k, v in _ACTION_TO_LABEL.items()
                    },
                    "forward_window": self._forward_window,
                    "buy_threshold": self._buy_threshold,
                    "sell_threshold": self._sell_threshold,
                },
                indent=2,
            )
        )
        logger.info("train.model_saved", path=str(path), meta=str(meta_path))

    @staticmethod
    def get_asset_id(symbol: str, venue: str = "bitso") -> int | None:
        """Obtiene el asset_id de la DB para el símbolo/venue dados."""
        with get_session() as session:
            asset = session.execute(
                select(Asset).where(
                    Asset.symbol == symbol,
                    Asset.venue == venue,
                )
            ).scalar_one_or_none()
            return asset.id if asset else None
