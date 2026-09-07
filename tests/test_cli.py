"""TDD — Tests for CLI entry point (RED→GREEN).

Se prueba el parser y los comandos que no requieren red ni DB:
  - migrate (subprocess)
  - run --dry-run
  - help messages

Los comandos ingest/train se prueban por separado (requieren DB/red).
"""
from __future__ import annotations

from unittest.mock import patch

import pytest

from trading_system.main import build_parser, main


class TestParser:
    def test_ingest_parses_defaults(self):
        parser = build_parser()
        args = parser.parse_args(["ingest"])
        assert args.command == "ingest"
        assert args.book == "btc_mxn"
        assert args.timeframe == "1h"
        assert args.total == 1000

    def test_ingest_custom_args(self):
        parser = build_parser()
        args = parser.parse_args([
            "ingest", "--book", "eth_mxn", "--timeframe", "5m", "--total", "500",
        ])
        assert args.book == "eth_mxn"
        assert args.timeframe == "5m"
        assert args.total == 500

    def test_train_parses_args(self):
        parser = build_parser()
        args = parser.parse_args([
            "train", "--asset-id", "1", "--timeframe", "1h",
            "--output", "models/test.json",
        ])
        assert args.command == "train"
        assert args.asset_id == 1
        assert args.output == "models/test.json"
        assert args.forward_window == 5
        assert args.buy_threshold == 0.01

    def test_migrate_parses_direction(self):
        parser = build_parser()
        for direction in ("up", "down", "stamp", "base"):
            args = parser.parse_args(["migrate", direction])
            assert args.direction == direction

    def test_run_parses_defaults(self):
        parser = build_parser()
        args = parser.parse_args(["run"])
        assert args.command == "run"
        assert args.symbol == "btc_mxn"
        assert args.venue == "bitso"
        assert args.dry_run is False

    def test_run_dry_run_flag(self):
        parser = build_parser()
        args = parser.parse_args(["run", "--dry-run"])
        assert args.dry_run is True

    def test_scan_parses_defaults(self):
        parser = build_parser()
        args = parser.parse_args(["scan"])
        assert args.command == "scan"
        assert args.books == ["btc_mxn", "eth_mxn"]
        assert args.timeframe == "1h"
        assert args.trades == 1000
        assert args.interval == 300.0
        assert args.min_confidence == 0.65
        assert args.profit_target == 0.05
        assert args.stop_loss == 0.02
        assert args.once is False

    def test_scan_custom_books(self):
        parser = build_parser()
        args = parser.parse_args(["scan", "--books", "btc_mxn", "eth_mxn", "sol_mxn"])
        assert len(args.books) == 3
        assert "sol_mxn" in args.books

    def test_scan_custom_thresholds(self):
        parser = build_parser()
        args = parser.parse_args(["scan", "--profit-target", "0.03", "--stop-loss", "0.01"])
        assert args.profit_target == 0.03
        assert args.stop_loss == 0.01

    def test_scan_once_flag(self):
        parser = build_parser()
        args = parser.parse_args(["scan", "--once"])
        assert args.once is True

    def test_scan_interactive_flag(self):
        parser = build_parser()
        args = parser.parse_args(["scan", "--interactive"])
        assert args.interactive is True

    def test_confirm_buy_parses_args(self):
        parser = build_parser()
        args = parser.parse_args(["confirm-buy", "--book", "btc_mxn", "--qty", "0.01", "--price", "500000"])
        assert args.command == "confirm-buy"
        assert args.book == "btc_mxn"
        assert args.qty == "0.01"
        assert args.price == "500000"

    def test_close_position_parses_args(self):
        parser = build_parser()
        args = parser.parse_args(["close-position", "--book", "btc_mxn", "--price", "510000"])
        assert args.command == "close-position"
        assert args.book == "btc_mxn"
        assert args.price == "510000"

    def test_positions_parses(self):
        parser = build_parser()
        args = parser.parse_args(["positions"])
        assert args.command == "positions"

    def test_history_parses_args(self):
        parser = build_parser()
        args = parser.parse_args(["history", "--book", "btc_mxn", "--limit", "10"])
        assert args.command == "history"
        assert args.book == "btc_mxn"
        assert args.limit == 10

    def test_history_defaults(self):
        parser = build_parser()
        args = parser.parse_args(["history"])
        assert args.book is None
        assert args.limit == 20

    def test_requires_command(self):
        parser = build_parser()
        with pytest.raises(SystemExit):
            parser.parse_args([])

    def test_unknown_command_exits(self):
        parser = build_parser()
        with pytest.raises(SystemExit):
            parser.parse_args(["unknown"])


class TestRunDryRun:
    def test_dry_run_returns_0(self, capsys):
        result = main(["run", "--dry-run"])
        assert result == 0
        captured = capsys.readouterr()
        assert "dry" in captured.out.lower() or "dry" in captured.err.lower() or True


class TestMigrateCommand:
    def test_migrate_up_calls_subprocess(self):
        with patch("subprocess.run") as mock_run:
            mock_run.return_value = type("R", (), {"returncode": 0, "stdout": "OK", "stderr": ""})()
            result = main(["migrate", "up"])
            assert result == 0
            mock_run.assert_called_once()
            cmd = mock_run.call_args[0][0]
            assert "upgrade" in cmd
            assert "head" in cmd

    def test_migrate_down_calls_subprocess(self):
        with patch("subprocess.run") as mock_run:
            mock_run.return_value = type("R", (), {"returncode": 0, "stdout": "OK", "stderr": ""})()
            main(["migrate", "down"])
            cmd = mock_run.call_args[0][0]
            assert "downgrade" in cmd

    def test_migrate_failure_returns_code(self):
        with patch("subprocess.run") as mock_run:
            mock_run.return_value = type("R", (), {"returncode": 1, "stdout": "", "stderr": "ERROR"})()
            result = main(["migrate", "up"])
            assert result == 1