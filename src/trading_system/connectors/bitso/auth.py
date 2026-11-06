"""
Autenticación para la API REST v3 de Bitso.

Bitso usa HMAC-SHA256 sobre el mensaje ``nonce + method + path + body``
firmado con el API secret como clave, y UN solo header Authorization::

    Authorization: Bitso <api_key>:<nonce>:<firma hex>

Nonce v2 (obligatorio desde nov-2025): timestamp Unix en milisegundos
(13 dígitos) + salt aleatorio de 6 dígitos (total 19 dígitos).

El ``path`` firmado debe coincidir EXACTAMENTE con el path del request
(incluye query string si lo hay) y el ``body`` con el JSON enviado.

Esta clase NO conoce HTTP ni websockets: solo construye headers. SRP
+ DIP: el conector REST la recibe por inyección.
"""
import hashlib
import hmac
import random
import time


class BitsoAuth:
    """Genera el header de autenticación para cada request a Bitso."""

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
        nonce = self._generate_nonce()
        message = f"{nonce}{method.upper()}{path}{body}"
        signature = hmac.new(
            self._api_secret.encode(),
            message.encode(),
            hashlib.sha256,
        ).hexdigest()
        return {
            "Authorization": f"Bitso {self._api_key}:{nonce}:{signature}",
        }

    @staticmethod
    def _generate_nonce() -> str:
        """Nonce v2: timestamp en ms (13 dígitos) + salt aleatorio (6)."""
        timestamp = int(time.time() * 1000)
        salt = random.randint(100000, 999999)
        return f"{timestamp}{salt}"

    @property
    def api_key(self) -> str:
        return self._api_key
