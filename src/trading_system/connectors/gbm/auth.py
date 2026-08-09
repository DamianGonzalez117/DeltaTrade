"""
Autenticación para la API de GBM.

GBM usa un modelo de token JWT:
  1. Login con api_key + api_secret → recibe token + TTL
  2. Requests incluyen ``Authorization: Bearer <token>``
  3. El token expira; ``ensure_valid_token`` refresca cuando es necesario

DIP: la llamada HTTP real se delega al conector REST mediante un
callback inyectado. ``GBMAuth`` no conoce ``httpx``.
"""
from collections.abc import Awaitable, Callable
from datetime import UTC, datetime, timedelta

LoginFn = Callable[[str, str], Awaitable[tuple[str, int]]]
_TOKEN_REFRESH_BUFFER = timedelta(seconds=60)


class GBMAuth:
    """Gestiona el ciclo de vida del token JWT de GBM."""

    def __init__(self, *, api_key: str, api_secret: str) -> None:
        if not api_key:
            raise ValueError("API key no puede estar vacía")
        if not api_secret:
            raise ValueError("API secret no puede estar vacío")
        self._api_key = api_key
        self._api_secret = api_secret
        self._token: str | None = None
        self._token_expires_at: datetime | None = None

    @property
    def token(self) -> str | None:
        return self._token

    async def login(self, login_fn: LoginFn) -> str:
        token, ttl_seconds = await login_fn(self._api_key, self._api_secret)
        self._token = token
        self._token_expires_at = datetime.now(UTC) + timedelta(seconds=ttl_seconds)
        return token

    def is_token_valid(self) -> bool:
        if self._token is None or self._token_expires_at is None:
            return False
        now = datetime.now(UTC)
        return self._token_expires_at - now > _TOKEN_REFRESH_BUFFER

    def build_headers(self) -> dict[str, str]:
        if self._token is None:
            raise RuntimeError("No hay token válido. Llama login() primero.")
        return {"Authorization": f"Bearer {self._token}"}

    async def ensure_valid_token(self, login_fn: LoginFn) -> None:
        if not self.is_token_valid():
            await self.login(login_fn)