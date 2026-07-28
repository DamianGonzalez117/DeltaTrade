"""
Configuración centralizada del sistema.

Toda la configuración se lee de variables de entorno (o de un archivo .env
en desarrollo). Ningún módulo debe leer os.environ directamente: siempre
deben importar `settings` desde este módulo, para tener una única fuente
de verdad y validación de tipos.
"""
from functools import lru_cache

from pydantic import Field
from pydantic_settings import BaseSettings, SettingsConfigDict


class Settings(BaseSettings):
    model_config = SettingsConfigDict(
        env_file=".env",
        env_file_encoding="utf-8",
        extra="ignore",
    )

    environment: str = Field(default="development", alias="ENVIRONMENT")
    log_level: str = Field(default="INFO", alias="LOG_LEVEL")

    # --- Base de datos ---
    database_url: str = Field(
        default="postgresql+psycopg://trading:trading@localhost:5432/trading_system",
        alias="DATABASE_URL",
    )
    db_echo: bool = Field(default=False, alias="DB_ECHO")

    # --- Redis ---
    redis_url: str = Field(default="redis://localhost:6379/0", alias="REDIS_URL")

    # --- Bitso ---
    bitso_api_key: str = Field(default="", alias="BITSO_API_KEY")
    bitso_api_secret: str = Field(default="", alias="BITSO_API_SECRET")
    bitso_ws_url: str = Field(default="wss://ws.bitso.com", alias="BITSO_WS_URL")
    bitso_rest_url: str = Field(default="https://api.bitso.com", alias="BITSO_REST_URL")

    # --- GBM ---
    gbm_api_key: str = Field(default="", alias="GBM_API_KEY")
    gbm_api_secret: str = Field(default="", alias="GBM_API_SECRET")
    gbm_base_url: str = Field(default="", alias="GBM_BASE_URL")

    # --- Notificaciones ---
    telegram_bot_token: str = Field(default="", alias="TELEGRAM_BOT_TOKEN")
    telegram_chat_id: str = Field(default="", alias="TELEGRAM_CHAT_ID")
    discord_webhook_url: str = Field(default="", alias="DISCORD_WEBHOOK_URL")

    @property
    def is_production(self) -> bool:
        return self.environment.lower() == "production"


@lru_cache
def get_settings() -> Settings:
    """Devuelve una instancia cacheada de Settings (patrón singleton)."""
    return Settings()


settings = get_settings()
