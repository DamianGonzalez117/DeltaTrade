"""
Configuración de reintentos con tenacity para conectores HTTP y WebSocket.

DIP: los conectores importan ``retry_async`` y lo aplican a sus métodos.
SRP: este módulo solo define la política de reintentos, no la lógica de negocio.

Política:
  - Reintenta en: timeouts, errores de red, HTTP 429 (rate limit), HTTP 5xx.
  - NO reintenta en: HTTP 4xx (excepto 429), errores de autenticación.
  - Backoff exponencial: 1s, 2s, 4s (máx 3 intentos).
"""
from __future__ import annotations

import httpx
import structlog
from tenacity import (
    retry,
    retry_if_exception_type,
    stop_after_attempt,
    wait_exponential,
)

logger = structlog.get_logger(__name__)

MAX_ATTEMPTS = 3
MIN_WAIT_S = 1
MAX_WAIT_S = 10


class TransientError(Exception):
    """Error transitorio que merece reintento (network, 5xx, 429)."""


def _is_transient_status(status_code: int) -> bool:
    """True si el código HTTP es retryable (429 o 5xx)."""
    return status_code == 429 or status_code >= 500


def to_transient_if_needed(exc: Exception) -> Exception:
    """Convierte errores httpx en TransientError si el status lo amerita."""
    if isinstance(exc, httpx.HTTPStatusError):
        status = exc.response.status_code
        if _is_transient_status(status):
            transient = TransientError(str(exc))
            transient.__cause__ = exc
            return transient
        return exc
    if isinstance(exc, (httpx.TimeoutException, httpx.NetworkError, httpx.ConnectError)):
        transient = TransientError(str(exc))
        transient.__cause__ = exc
        return transient
    return exc


async def retry_async(coro_fn, *args, **kwargs):
    """Ejecuta una corrutina con política de reintentos transitorios.

    Usage::

        data = await retry_async(self._request, "GET", "/v3/balance/")
    """

    @retry(
        stop=stop_after_attempt(MAX_ATTEMPTS),
        wait=wait_exponential(multiplier=1, min=MIN_WAIT_S, max=MAX_WAIT_S),
        retry=retry_if_exception_type(TransientError),
        reraise=True,
    )
    async def _attempt():
        try:
            return await coro_fn(*args, **kwargs)
        except Exception as exc:
            transient = to_transient_if_needed(exc)
            if isinstance(transient, TransientError):
                logger.warning("retry.transient_error", error=str(transient))
                raise transient
            raise

    return await _attempt()