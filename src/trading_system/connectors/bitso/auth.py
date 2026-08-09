"""
Autenticación para la API REST v3 de Bitso.

Bitso usa HMAC-SHA256 sobre el mensaje ``nonce + method + path + body``
firma con el API secret como clave. Los headers resultantes son::

    Authorization: Bitso <api_key>
    X-Request-Timestamp: <nonce>
    X-Request-Signature: <firma hex>

Esta clase NO conoce HTTP ni websockets: solo construye headers. SRP
+ DIP: el conector REST la recibe por inyección.
"""
import hashlib
import hmac
import time


class BitsoAuth:
    """Genera los headers de autenticación para cada request a Bitso."""

    def __init__(self, api_key: str, api_secret: str) -> None:
        if not api_key:
            raise ValueError("API key no puede estar vacía")
        if not api_secret:
            raise ValueError("API secret no puede estar vacío")
        self._api_key = api_key
        self._api_secret = api_secret

    def build_headers(
        self,
        method: str,
        path: str,
        body: str = "",
    ) -> dict[str, str]:
        nonce = str(int(time.time() * 1000))
        message = f"{nonce}{method.upper()}{path}{body}"
        signature = hmac.new(
            self._api_secret.encode(),
            message.encode(),
            hashlib.sha256,
        ).hexdigest()
        return {
            "Authorization": f"Bitso {self._api_key}",
            "X-Request-Timestamp": nonce,
            "X-Request-Signature": signature,
        }

    @property
    def api_key(self) -> str:
        return self._api_key