"""Conversión de texto con marcadores simples a HTML seguro para Telegram.

El Markdown legacy de Telegram (parse_mode="Markdown") rompe con `_`
presente en nombres de libros (btc_mxn), paréntesis y otros caracteres:
la API responde 400 "can't parse entities" y el mensaje nunca llega.

HTML es robusto: se escapan todos los caracteres especiales y solo se
convierten marcadores explícitos:

    *texto*   → <b>texto</b>
    `texto`   → <code>texto</code>
    _texto_   → <i>texto</i> (solo con `_` en bordes de palabra)
"""
from __future__ import annotations

import html
import re

_BOLD = re.compile(r"\*([^*\n]+)\*")
_CODE = re.compile(r"`([^`\n]+)`")
_ITALIC = re.compile(r"(?<![\w*`])_([^_\n]+)_(?![\w*`])")


def to_telegram_html(text: str) -> str:
    """Escapa HTML y convierte marcadores *bold*, `code` y _italic_ a tags."""
    escaped = html.escape(text)
    escaped = _BOLD.sub(r"<b>\1</b>", escaped)
    escaped = _CODE.sub(r"<code>\1</code>", escaped)
    escaped = _ITALIC.sub(r"<i>\1</i>", escaped)
    return escaped
