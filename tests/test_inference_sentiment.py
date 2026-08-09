"""TDD — Tests for SentimentAnalyzer ABC and FinBERTSentimentAnalyzer (RED phase).

``SentimentAnalyzer`` es el contrato: dado lista de headlines, devuelve
un score de sentimiento normalizado [-1, 1].

``FinBERTSentimentAnalyzer`` delega al pipeline de HuggingFace, que se
inyecta por constructor (DIP) para tests.
"""
from unittest.mock import MagicMock

import pytest

from trading_system.inference.sentiment.base import SentimentAnalyzer, SentimentResult
from trading_system.inference.sentiment.finbert import FinBERTSentimentAnalyzer


class TestSentimentAnalyzerABC:
    def test_cannot_instantiate_directly(self):
        with pytest.raises(TypeError):
            SentimentAnalyzer()  # type: ignore[abstract]


class TestFinBERTSentimentAnalyzer:
    def test_analyze_positive_headlines_returns_positive_score(self):
        mock_pipeline = MagicMock()
        mock_pipeline.return_value = [
            [{"label": "positive", "score": 0.85}],
        ]
        analyzer = FinBERTSentimentAnalyzer(pipeline=mock_pipeline)
        result = analyzer.analyze(["Bitcoin surges on ETF approval"])

        assert isinstance(result, SentimentResult)
        assert result.score > 0
        assert result.label == "positive"

    def test_analyze_negative_headlines_returns_negative_score(self):
        mock_pipeline = MagicMock()
        mock_pipeline.return_value = [
            [{"label": "negative", "score": 0.90}],
        ]
        analyzer = FinBERTSentimentAnalyzer(pipeline=mock_pipeline)
        result = analyzer.analyze(["Market crashes amid recession fears"])

        assert result.score < 0
        assert result.label == "negative"

    def test_analyze_neutral_headlines_returns_near_zero(self):
        mock_pipeline = MagicMock()
        mock_pipeline.return_value = [
            [{"label": "neutral", "score": 0.75}],
        ]
        analyzer = FinBERTSentimentAnalyzer(pipeline=mock_pipeline)
        result = analyzer.analyze(["Fed holds rates unchanged"])

        assert abs(result.score) < 0.15
        assert result.label == "neutral"

    def test_analyze_multiple_headlines_averages_scores(self):
        mock_pipeline = MagicMock()
        mock_pipeline.return_value = [
            [{"label": "positive", "score": 0.80}],
            [{"label": "negative", "score": 0.70}],
        ]
        analyzer = FinBERTSentimentAnalyzer(pipeline=mock_pipeline)
        result = analyzer.analyze(["Good news", "Bad news"])

        assert -0.2 < result.score < 0.2

    def test_analyze_empty_headlines_returns_neutral(self):
        mock_pipeline = MagicMock()
        analyzer = FinBERTSentimentAnalyzer(pipeline=mock_pipeline)
        result = analyzer.analyze([])

        assert result.score == 0
        assert result.label == "neutral"
        mock_pipeline.assert_not_called()

    def test_model_name_property(self):
        analyzer = FinBERTSentimentAnalyzer(pipeline=MagicMock())
        assert "finbert" in analyzer.model_name.lower()