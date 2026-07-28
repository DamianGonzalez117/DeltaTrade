# Sistema de Trading Multi-Mercado (Bitso + GBM)

Sistema de trading algorítmico que combina modelos cuantitativos (XGBoost/LSTM),
análisis de sentimiento (FinBERT/LLMs) y un motor de riesgo determinista, para
operar simultáneamente activos digitales (Bitso, 24/7) y renta variable/ETFs
(GBM, horario bursátil).

## Estado actual del proyecto

✅ **Fase 0 — Fundaciones (completada en esta sesión)**
- Estructura del repo y `pyproject.toml`
- Esquema de base de datos (PostgreSQL) con SQLAlchemy 2.0 + Alembic
- Enums de dominio compartidos entre capas
- Interfaz abstracta de conectores (`ExchangeConnector`, `StreamingMarketDataConnector`, `PollingMarketDataConnector`)
- Docker Compose (Postgres + Redis) para desarrollo local

⬜ Fase 1 — Conector Bitso (WebSocket + REST)
⬜ Fase 2 — Conector GBM (polling + calendario bursátil)
⬜ Fase 3 — Pipeline de features técnicos (RSI/MACD/ATR)
⬜ Fase 4 — Motor de riesgo determinista
⬜ Fase 5 — Motor de inferencia (XGBoost/FinBERT)
⬜ Fase 6 — Capa de ejecución y notificaciones

## Arquitectura

```
[ Capa de Datos ]        --->   [ Capa de Inteligencia (IA) ]
• WebSockets Bitso              • Modelos ML/DL (XGBoost / TFT)
• REST Polling GBM              • NLP / Sentimiento (FinBERT)
• Feed Noticias / RSS           • Agentes RL (Stable-Baselines3)
                                            │
                                            ▼
[ Capa de Ejecución ]    <---   [ Motor de Gestión de Riesgo ]
• Conector Bitso (Official)     • Control de Drawdown y Position Sizing
• Conector GBM (API/Community)  • Verificación de Horarios y Límites
```

## Estructura del repositorio

```
trading_system/
├── docker-compose.yml          # Postgres + Redis para desarrollo
├── pyproject.toml              # dependencias y metadata del paquete
├── .env.example                # variables de entorno de referencia
├── alembic.ini / alembic/      # migraciones de base de datos
├── src/trading_system/
│   ├── config.py               # Settings centralizados (pydantic-settings)
│   ├── core/
│   │   └── enums.py            # enums de dominio compartidos
│   ├── db/
│   │   ├── base.py             # Base declarativa + convención de nombres
│   │   ├── session.py          # engine + sessionmaker
│   │   └── models/
│   │       ├── market.py       # Asset, MarketDataOHLCV
│   │       ├── signals.py      # SentimentScore, AISignal
│   │       ├── risk.py         # RiskEvaluation
│   │       ├── trading.py      # Order, Fill, Position, PortfolioSnapshot
│   │       └── notifications.py# NotificationLog
│   ├── connectors/
│   │   ├── base.py             # interfaces abstractas (ExchangeConnector, etc.)
│   │   ├── bitso/              # (Fase 1)
│   │   └── gbm/                # (Fase 2)
│   ├── risk/                   # (Fase 4)
│   └── services/                # orquestación entre capas
└── tests/
    └── test_schema.py          # smoke test del esquema completo
```

## Modelo de datos (resumen)

| Tabla                 | Capa         | Propósito                                                             |
|------------------------|--------------|------------------------------------------------------------------------|
| `assets`               | Datos        | Catálogo único de instrumentos (símbolo + venue + clase de activo)     |
| `market_data_ohlcv`    | Datos        | Velas de precio por activo/timeframe                                   |
| `sentiment_scores`     | Inteligencia | Puntuación de sentimiento por noticia/titular (FinBERT/LLM)            |
| `ai_signals`           | Inteligencia | Señal `buy/sell/hold` + confianza, con snapshot de features usados     |
| `risk_evaluations`     | Riesgo       | Decisión determinista sobre cada señal, ANTES de ejecutar nada         |
| `orders`               | Ejecución    | Órdenes enviadas al exchange, siempre ligadas a un `risk_evaluations`  |
| `fills`                | Ejecución    | Ejecuciones parciales/totales de una orden                              |
| `positions`            | Ejecución    | Posiciones abiertas/cerradas por activo                                 |
| `portfolio_snapshots`  | Ejecución    | Foto periódica de equity/drawdown del portafolio                        |
| `notifications_log`    | Transversal  | Auditoría de alertas enviadas (Telegram/Discord/logs)                  |

**Regla de negocio clave ya reflejada en el esquema:** una orden (`orders`)
solo puede crearse con un `risk_evaluation_id` que apunte a una evaluación con
`decision = approved` o `adjusted`. Esto materializa en la base de datos el
objetivo de "interponer un motor de reglas determinista antes de cualquier
interacción con las APIs de ejecución".

## Cómo arrancar en local

```bash
# 1. Levantar Postgres + Redis
docker compose up -d

# 2. Crear entorno virtual e instalar dependencias
python3.11 -m venv .venv
source .venv/bin/activate
pip install -e ".[dev]"

# 3. Copiar variables de entorno
cp .env.example .env

# 4. Generar la primera migración a partir de los modelos
alembic revision --autogenerate -m "esquema inicial"
alembic upgrade head

# 5. Correr el smoke test del esquema
pytest tests/test_schema.py -v
```

## Próximo paso sugerido

Fase 1: implementar `BitsoWebSocketConnector` (streaming de `btc_mxn`/`eth_mxn`)
sobre la interfaz `StreamingMarketDataConnector`, y `BitsoRESTConnector` sobre
`ExchangeConnector`, con reintentos (`tenacity`) y reconexión automática del
socket.
