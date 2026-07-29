"""Conectores oficiales de Bitso (API v3).

Exporta:
  - ``BitsoAuth``: firma HMAC-SHA256 de requests.
  - ``BitsoRESTConnector``: ejecución de órdenes y balances (REST).
  - ``BitsoWebSocketConnector``: streaming de ticker en tiempo real.
"""
from trading_system.connectors.bitso.auth import BitsoAuth
from trading_system.connectors.bitso.rest import BitsoRESTConnector
from trading_system.connectors.bitso.websocket import BitsoWebSocketConnector

__all__ = [
    "BitsoAuth",
    "BitsoRESTConnector",
    "BitsoWebSocketConnector",
]