"""
Bot interactivo de Telegram: recibe comandos del usuario via long polling.

Comandos soportados:
    compré <book> <qty> <precio>     → Abre posición
    vendí <book> <precio>            → Cierra posición y calcula P&L
    posiciones                       → Lista posiciones abiertas
    historial [book] [n]             → Muestra historial de señales
    estado                           → Muestra últimas señales del scan
    ayuda                            → Lista de comandos

DIP: ``SignalHistoryService`` y ``MarketMonitor`` se inyectan.
SRP: solo parsea y despacha comandos. No escanea por sí mismo.
"""
from __future__ import annotations

import json
from decimal import Decimal, InvalidOperation
from typing import TYPE_CHECKING

import httpx
import structlog

if TYPE_CHECKING:
    from trading_system.notifications.dispatcher import NotificationDispatcher
    from trading_system.services.monitor import MarketMonitor
    from trading_system.services.signal_history import SignalHistoryService
    from trading_system.services.trade_analyzer import TradeAnalyzer

logger = structlog.get_logger(__name__)

_API_BASE = "https://api.telegram.org/bot"
_GET_UPDATES = "/getUpdates"
_SEND_MESSAGE = "/sendMessage"
_LONG_POLL_TIMEOUT = 30


class TelegramCommandBot:
    """Bot que recibe mensajes de Telegram y ejecuta comandos de trading."""

    def __init__(
        self,
        *,
        bot_token: str,
        chat_id: str,
        history_service: SignalHistoryService,
        dispatcher: NotificationDispatcher,
        monitor: MarketMonitor | None = None,
        trade_analyzer: TradeAnalyzer | None = None,
        client: httpx.AsyncClient | None = None,
    ) -> None:
        self._bot_token = bot_token
        self._chat_id = chat_id
        self._history = history_service
        self._dispatcher = dispatcher
        self._monitor = monitor
        self._analyzer = trade_analyzer
        if client is not None:
            self._client = client
            self._owns_client = False
        else:
            self._client = httpx.AsyncClient(timeout=httpx.Timeout(60.0))
            self._owns_client = True
        self._offset = 0
        self._running = False

    async def run(self) -> None:
        """Loop de long polling. Bloquea hasta stop()."""
        self._running = True
        logger.info("bot.starting", chat_id=self._chat_id)

        await self._send_message(
            "🤖 *Bot activo*\nEnvía _ayuda_ para ver los comandos disponibles."
        )

        while self._running:
            try:
                updates = await self._poll_updates()
                for update in updates:
                    await self._handle_update(update)
            except httpx.TimeoutException:
                continue
            except (httpx.NetworkError, httpx.HTTPError) as exc:
                logger.warning("bot.poll_error", error=str(exc))
                import asyncio
                await asyncio.sleep(5)

        logger.info("bot.stopped")

    def stop(self) -> None:
        self._running = False

    async def _poll_updates(self) -> list[dict]:
        url = f"{_API_BASE}{self._bot_token}{_GET_UPDATES}"
        params = {
            "offset": str(self._offset),
            "timeout": str(_LONG_POLL_TIMEOUT),
        }
        response = await self._client.get(url, params=params)
        response.raise_for_status()
        data = response.json()
        updates = data.get("result", [])
        if updates:
            last_update_id = updates[-1]["update_id"]
            self._offset = last_update_id + 1
        return updates

    async def _handle_update(self, update: dict) -> None:
        message = update.get("message")
        if message is None:
            return

        chat_id = str(message.get("chat", {}).get("id", ""))
        if chat_id != self._chat_id:
            logger.warning("bot.unauthorized_chat", chat_id=chat_id)
            return

        text = (message.get("text") or "").strip()
        if not text:
            return

        logger.info("bot.received", text=text)
        await self._dispatch_command(text)

    async def _dispatch_command(self, text: str) -> None:
        parts = text.lower().split()
        if not parts:
            return

        cmd = parts[0]

        if cmd in ("compré", "compre", "buy", "c"):
            await self._cmd_buy(parts, text)
        elif cmd in ("vendí", "vendi", "sell", "v"):
            await self._cmd_sell(parts, text)
        elif cmd in ("posiciones", "positions", "p"):
            await self._cmd_positions()
        elif cmd in ("historial", "history", "h"):
            await self._cmd_history(parts)
        elif cmd in ("estado", "status", "s"):
            await self._cmd_status()
        elif cmd in ("ayuda", "help", "?"):
            await self._cmd_help()
        else:
            await self._send_message(
                f"Comando no reconocido: _{text}_\nEnvía _ayuda_ para ver los comandos."
            )

    async def _cmd_buy(self, parts: list[str], original: str) -> None:
        try:
            book = parts[1]
            qty = Decimal(parts[2])
            price = Decimal(parts[3])
        except (IndexError, InvalidOperation):
            await self._send_message(
                "Formato: `compré <book> <qty> <precio>`\n"
                "Ej: `compré btc_mxn 0.01 505000`"
            )
            return

        try:
            position = self._history.confirm_buy(book, qty, price)
            body = (
                f"✅ *Compra registrada*\n"
                f"Libro: {book}\n"
                f"Cantidad: {position.quantity}\n"
                f"Precio entrada: ${position.avg_entry_price:,.2f}\n"
                f"ID posición: {position.id}"
            )
            body += self._target_line(book, float(position.avg_entry_price))
            body += "\n\nTe avisaré cuando sea momento de vender."
            await self._send_message(body)
        except ValueError as exc:
            await self._send_message(f"❌ {exc}")

    async def _cmd_sell(self, parts: list[str], original: str) -> None:
        try:
            book = parts[1]
            price = Decimal(parts[2])
        except (IndexError, InvalidOperation):
            await self._send_message(
                "Formato: `vendí <book> <precio>`\n"
                "Ej: `vendí btc_mxn 520000`"
            )
            return

        try:
            position = self._history.close_position(book, price)
            pnl = position.realized_pnl
            if pnl >= 0:
                pnl_str = f"+${pnl:,.2f} 🟢"
            else:
                pnl_str = f"-${abs(pnl):,.2f} 🔴"

            body = (
                f"✅ *Venta registrada*\n"
                f"Libro: {book}\n"
                f"Precio salida: ${Decimal(str(price)):,.2f}\n"
                f"P&L: {pnl_str}\n"
                f"ID posición: {position.id}"
            )
            body += self._stats_line(book)
            await self._send_message(body)
        except ValueError as exc:
            await self._send_message(f"❌ {exc}")

    def _target_line(self, book: str, entry_price: float) -> str:
        """Línea con el objetivo de venta sugerido para la posición recién abierta."""
        if self._analyzer is None:
            return ""
        try:
            target_pct = self._analyzer.suggest_profit_target_pct(book, default_pct=5.0)
            target_price = entry_price * (1 + target_pct / 100)
            return (
                f"\n🎯 *Objetivo de venta:* ${target_price:,.2f} (+{target_pct:.1f}%)"
            )
        except (ValueError, RuntimeError, OSError):
            return ""

    def _stats_line(self, book: str) -> str:
        """Línea con stats del usuario y próximo objetivo tras cerrar posición."""
        if self._analyzer is None:
            return ""
        try:
            parts: list[str] = []
            stats = self._analyzer.get_closed_stats(book)
            if stats is not None:
                parts.append(
                    f"\n📈 *Tu historial:* {stats.n_trades} trades, "
                    f"{stats.win_rate * 100:.0f}% win-rate, {stats.avg_profit_pct:+.1f}% promedio"
                )
            target_pct = self._analyzer.suggest_profit_target_pct(book, default_pct=5.0)
            parts.append(f"\n🎯 *Próximo objetivo sugerido:* +{target_pct:.1f}%")
            return "".join(parts)
        except (ValueError, RuntimeError, OSError):
            logger.warning("bot.analyzer_failed", book=book)
            return ""

    async def _cmd_positions(self) -> None:
        positions = self._history.get_all_open_positions()
        if not positions:
            await self._send_message("📭 No tienes posiciones abiertas.")
            return

        lines = [f"📊 *Posiciones abiertas ({len(positions)}):*"]
        for pos in positions:
            lines.append(
                f"\n• {pos.asset.symbol}\n"
                f"  Cantidad: {pos.quantity}\n"
                f"  Entrada: ${pos.avg_entry_price:,.2f}\n"
                f"  Abierta: {pos.opened_at.strftime('%Y-%m-%d %H:%M')}"
            )
        await self._send_message("\n".join(lines))

    async def _cmd_history(self, parts: list[str]) -> None:
        book = parts[1] if len(parts) > 1 else None
        limit = 10
        if len(parts) > 2:
            try:
                limit = int(parts[2])
            except ValueError:
                pass

        signals = self._history.get_signal_history(book=book, limit=limit)
        if not signals:
            await self._send_message("📭 No hay señales registradas.")
            return

        lines = [f"📋 *Historial ({len(signals)}):*"]
        for sig in signals:
            action_str = sig.action.value if hasattr(sig.action, "value") else str(sig.action)
            lines.append(
                f"\n• {sig.generated_at.strftime('%m-%d %H:%M')} "
                f"{sig.asset.symbol:10s} {action_str.upper():4s} "
                f"conf={sig.confidence:.2f}"
            )
        await self._send_message("\n".join(lines))

    async def _cmd_status(self) -> None:
        if self._monitor is None or not self._monitor.last_signals:
            await self._send_message("📡 El monitor no ha escaneado todavía.")
            return

        lines = ["📡 *Últimas señales:*"]
        for book, signal in self._monitor.last_signals.items():
            pos = self._history.get_open_position(book)
            pos_str = " 📌" if pos else ""
            lines.append(
                f"\n• {book:10s} {signal.action.value.upper():4s} "
                f"RSI={signal.rsi:.1f} conf={float(signal.confidence):.0%}{pos_str}"
            )
        await self._send_message("\n".join(lines))

    async def _cmd_help(self) -> None:
        help_text = (
            "🤖 *Comandos disponibles:*\n\n"
            "• `compré btc_mxn 0.01 505000`\n"
            "  Registra una compra y abre posición\n\n"
            "• `vendí btc_mxn 520000`\n"
            "  Cierra posición y calcula P&L\n\n"
            "• `posiciones`\n"
            "  Lista tus posiciones abiertas\n\n"
            "• `historial [book] [n]`\n"
            "  Muestra últimas señales (ej: historial btc_mxn 5)\n\n"
            "• `estado`\n"
            "  Muestra últimas señales del scan\n\n"
            "• `ayuda`\n"
            "  Muestra esta lista"
        )
        await self._send_message(help_text)

    async def _send_message(self, text: str) -> None:
        from trading_system.notifications.formatting import to_telegram_html

        url = f"{_API_BASE}{self._bot_token}{_SEND_MESSAGE}"
        payload = json.dumps({
            "chat_id": self._chat_id,
            "text": to_telegram_html(text),
            "parse_mode": "HTML",
        })
        try:
            response = await self._client.post(
                url,
                content=payload,
                headers={"Content-Type": "application/json"},
            )
            if response.status_code != 200:
                logger.error(
                    "bot.api_error",
                    status=response.status_code,
                    detail=response.text[:300],
                )
        except httpx.HTTPError as exc:
            logger.error("bot.send_failed", error=str(exc))

    async def aclose(self) -> None:
        if self._owns_client:
            await self._client.aclose()
