"""TDD — Tests for Notifier ABC, LogNotifier, TelegramNotifier, DiscordNotifier (RED phase).

``Notifier`` es el contrato: dado un ``NotificationMessage``, lo envía
al canal correspondiente.

DIP: cada notifier recibe su cliente HTTP por constructor (tests con MockTransport).
SRP: cada notifier solo sabe enviar a un canal.
"""

import httpx
import pytest

from trading_system.core.enums import NotificationChannel, NotificationLevel
from trading_system.notifications.base import NotificationMessage, Notifier
from trading_system.notifications.discord import DiscordNotifier
from trading_system.notifications.log_notifier import LogNotifier
from trading_system.notifications.telegram import TelegramNotifier

# ---------------------------------------------------------------------------
# Fixture
# ---------------------------------------------------------------------------

@pytest.fixture
def sample_message() -> NotificationMessage:
    return NotificationMessage(
        level=NotificationLevel.INFO,
        title="Order Filled",
        body=" BUY 0.01 BTC @ 500000 MXN on bitso",
        channel=NotificationChannel.LOG,
        related_entity_type="order",
        related_entity_id=42,
    )


# ---------------------------------------------------------------------------
# Notifier ABC
# ---------------------------------------------------------------------------

class TestNotifierABC:
    def test_cannot_instantiate_directly(self):
        with pytest.raises(TypeError):
            Notifier()  # type: ignore[abstract]


# ---------------------------------------------------------------------------
# LogNotifier
# ---------------------------------------------------------------------------

class TestLogNotifier:
    async def test_send_does_not_raise(self, sample_message):
        notifier = LogNotifier()
        await notifier.send(sample_message)

    def test_channel_is_log(self):
        notifier = LogNotifier()
        assert notifier.channel == NotificationChannel.LOG

    async def test_send_critical_message_does_not_raise(self):
        msg = NotificationMessage(
            level=NotificationLevel.CRITICAL,
            title="Drawdown Exceeded",
            body="Current drawdown 25% exceeds max 20%",
            channel=NotificationChannel.LOG,
        )
        notifier = LogNotifier()
        await notifier.send(msg)


# ---------------------------------------------------------------------------
# TelegramNotifier
# ---------------------------------------------------------------------------

class TestToTelegramHtml:
    """Marcadores simples → HTML seguro (fix: Markdown rompe con btc_mxn)."""

    def test_bold_marker_converted(self):
        from trading_system.notifications.formatting import to_telegram_html
        assert to_telegram_html("*BUY* ahora") == "<b>BUY</b> ahora"

    def test_code_marker_converted(self):
        from trading_system.notifications.formatting import to_telegram_html
        assert to_telegram_html("usa `vendí btc_mxn`") == "usa <code>vendí btc_mxn</code>"

    def test_italic_marker_converted(self):
        from trading_system.notifications.formatting import to_telegram_html
        assert to_telegram_html("Envía _ayuda_ ahora") == "Envía <i>ayuda</i> ahora"

    def test_underscore_in_book_names_stays_literal(self):
        from trading_system.notifications.formatting import to_telegram_html
        out = to_telegram_html("Libro: btc_mxn")
        assert out == "Libro: btc_mxn"
        assert "_" in out

    def test_two_books_not_converted_to_italic(self):
        from trading_system.notifications.formatting import to_telegram_html
        out = to_telegram_html("btc_mxn eth_mxn")
        assert out == "btc_mxn eth_mxn"  # _ rodeado de word chars no es italic

    def test_escapes_html_special_chars(self):
        from trading_system.notifications.formatting import to_telegram_html
        out = to_telegram_html("P&L < 0 > entra")
        assert out == "P&amp;L &lt; 0 &gt; entra"

    def test_unbalanced_markers_stay_literal(self):
        from trading_system.notifications.formatting import to_telegram_html
        out = to_telegram_html("precio * 5 y `comando")
        assert out == "precio * 5 y `comando"


class TestTelegramNotifier:
    def _make_notifier(self, handler) -> TelegramNotifier:
        transport = httpx.MockTransport(handler)
        client = httpx.AsyncClient(transport=transport)
        return TelegramNotifier(
            bot_token="123:abc",
            chat_id="999",
            client=client,
        )

    async def test_sends_post_to_telegram_api(self, sample_message):
        seen = {}

        def handler(request: httpx.Request) -> httpx.Response:
            seen["url"] = str(request.url)
            seen["method"] = request.method
            return httpx.Response(200, json={"ok": True})

        notifier = self._make_notifier(handler)
        await notifier.send(sample_message)
        await notifier.aclose()

        assert "api.telegram.org" in seen["url"]
        assert "123:abc" in seen["url"]
        assert seen["method"] == "POST"

    async def test_sends_message_text_in_body(self, sample_message):
        seen_body = {}

        def handler(request: httpx.Request) -> httpx.Response:
            import json
            seen_body["json"] = json.loads(request.content)
            return httpx.Response(200, json={"ok": True})

        notifier = self._make_notifier(handler)
        await notifier.send(sample_message)
        await notifier.aclose()

        assert seen_body["json"]["chat_id"] == "999"
        assert "Order Filled" in seen_body["json"]["text"]

    async def test_uses_html_parse_mode(self, sample_message):
        seen_body = {}

        def handler(request: httpx.Request) -> httpx.Response:
            import json
            seen_body["json"] = json.loads(request.content)
            return httpx.Response(200, json={"ok": True})

        notifier = self._make_notifier(handler)
        await notifier.send(sample_message)
        await notifier.aclose()

        assert seen_body["json"]["parse_mode"] == "HTML"
        assert "<b>" in seen_body["json"]["text"]  # título convertido a bold

    async def test_book_underscore_survives_html_mode(self, sample_message):
        seen_body = {}

        def handler(request: httpx.Request) -> httpx.Response:
            import json
            seen_body["json"] = json.loads(request.content)
            return httpx.Response(200, json={"ok": True})

        message = NotificationMessage(
            level=NotificationLevel.INFO,
            title="Señal BUY — btc_mxn",
            body="🟢 *BUY* btc_mxn\nObjetivo: +5%",
        )
        notifier = self._make_notifier(handler)
        result = await notifier.send(message)
        await notifier.aclose()

        assert result is True
        assert "btc_mxn" in seen_body["json"]["text"]

    async def test_api_error_is_logged_and_returns_false(self, sample_message):
        seen_body = {}

        def handler(request: httpx.Request) -> httpx.Response:
            return httpx.Response(400, json={
                "ok": False,
                "description": "Bad Request: can't parse entities",
            })

        notifier = self._make_notifier(handler)
        result = await notifier.send(sample_message)
        await notifier.aclose()

        assert result is False
        assert seen_body == {}  # handler no captura nada; el log va por structlog

    async def test_http_error_does_not_raise_but_returns_false(self, sample_message):
        def handler(request: httpx.Request) -> httpx.Response:
            return httpx.Response(500, json={"ok": False})

        notifier = self._make_notifier(handler)
        result = await notifier.send(sample_message)
        await notifier.aclose()
        assert result is False

    def test_channel_is_telegram(self):
        notifier = TelegramNotifier(bot_token="x", chat_id="y")
        assert notifier.channel == NotificationChannel.TELEGRAM


# ---------------------------------------------------------------------------
# DiscordNotifier
# ---------------------------------------------------------------------------

class TestDiscordNotifier:
    def _make_notifier(self, handler) -> DiscordNotifier:
        transport = httpx.MockTransport(handler)
        client = httpx.AsyncClient(transport=transport)
        return DiscordNotifier(
            webhook_url="https://discord.com/api/webhooks/123/abc",
            client=client,
        )

    async def test_sends_post_to_webhook(self, sample_message):
        seen = {}

        def handler(request: httpx.Request) -> httpx.Response:
            seen["url"] = str(request.url)
            seen["method"] = request.method
            return httpx.Response(204)

        notifier = self._make_notifier(handler)
        await notifier.send(sample_message)
        await notifier.aclose()

        assert "discord.com" in seen["url"]
        assert seen["method"] == "POST"

    async def test_sends_embed_with_title_and_description(self, sample_message):
        seen_body = {}

        def handler(request: httpx.Request) -> httpx.Response:
            import json
            seen_body["json"] = json.loads(request.content)
            return httpx.Response(204)

        notifier = self._make_notifier(handler)
        await notifier.send(sample_message)
        await notifier.aclose()

        assert "embeds" in seen_body["json"]
        embed = seen_body["json"]["embeds"][0]
        assert embed["title"] == "Order Filled"

    async def test_http_error_returns_false(self, sample_message):
        def handler(request: httpx.Request) -> httpx.Response:
            return httpx.Response(400, json={"error": "bad webhook"})

        notifier = self._make_notifier(handler)
        result = await notifier.send(sample_message)
        await notifier.aclose()
        assert result is False

    def test_channel_is_discord(self):
        notifier = DiscordNotifier(webhook_url="https://discord.com/x")
        assert notifier.channel == NotificationChannel.DISCORD