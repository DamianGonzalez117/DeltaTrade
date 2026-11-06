# Sistema de Trading Multi-Mercado (Bitso + GBM)

Sistema de trading algorítmico que combina modelos cuantitativos (XGBoost/LSTM),
análisis de sentimiento (FinBERT/LLMs) y un motor de riesgo determinista, para
operar simultáneamente activos digitales (Bitso, 24/7) y renta variable/ETFs
(GBM, horario bursátil).

## Estado actual del proyecto

✅ **Fase 0 — Fundaciones**
- Estructura del repo y `pyproject.toml`
- Esquema de base de datos (PostgreSQL) con SQLAlchemy 2.0 + Alembic
- Enums de dominio compartidos entre capas
- Interfaz abstracta de conectores (`ExchangeConnector`, `StreamingMarketDataConnector`, `PollingMarketDataConnector`)
- Docker Compose (Postgres + Redis) para desarrollo local

✅ **Fase 1 — Conector Bitso (WebSocket + REST)** — auth HMAC v3 (nonce v2), balances, órdenes y fills
✅ **Fase 3 — Pipeline de features técnicos (RSI/MACD/ATR)**
✅ **Fase 6 (parcial) — Asistente de trading conversacional (Telegram + Bitso)**

⬜ Fase 2 — Conector GBM (polling + calendario bursátil)
⬜ Fase 4 — Motor de riesgo determinista (existe, falta conectar al flujo live)
⬜ Fase 5 — Motor de inferencia (XGBoost/FinBERT) en el flujo live

## 🫱 Bot asistente de trading (Telegram + Bitso)

El comando `scan --interactive` levanta dos tareas: el monitor de mercado
(loop de escaneo) y el bot de Telegram (long polling). Cuando hay llaves de
Bitso (`BITSO_API_KEY`/`BITSO_API_SECRET` en `.env`), las confirmaciones
ejecutan **órdenes reales de mercado**; sin llaves (o con `--paper`) todo se
simula (modo papel).

```bash
python -m trading_system.main scan --interactive            # live con keys
python -m trading_system.main scan --interactive --paper    # simulado
```

### Flujo de compra

1. El monitor detecta una señal BUY (confianza ≥ `--min-confidence`, sin
   posición abierta en ese libro y sin propuesta pendiente).
2. Te llega un mensaje con la **criptomoneda**, **% de éxito**, precio,
   objetivo de venta, **tiempo estimado hasta la venta** y tu **saldo
   disponible** en MXN.
3. Te pregunta *¿cuánto quieres comprar?* → respondes el monto en MXN
   (`500`, `$1,500.50 mxn`), `todo` para usar todo el saldo o `cancelar`.
4. Se valida el saldo contra la API, se coloca una **orden de mercado por
   el monto en MXN** (`minor`) y se registra la posición con la cantidad y
   el precio **real** de los fills (`GET /v3/order_trades/`).

### Flujo de venta

1. Con posición abierta, se te pide confirmación cuando: se alcanza el
   objetivo de ganancia (`--profit-target`), se cruza el stop loss
   (`--stop-loss`) o aparece una señal técnica de venta.
2. Respondes **sí** → se vende toda la posición (orden de mercado) y te
   llega la **ganancia neta** (comisiones incluidas) y el **tiempo desde
   la compra**.
3. Respondes **no** → te llega una **estadística de qué pasaría si esperas**
   (casos históricos con RSI similar: % que subieron/bajaron, ganancia y
   pérdida promedio, mejor/peor caso, tendencia y recomendación; el
   horizonte es adaptativo: mira más lejos mientras más historial haya,
   de 6 hasta 48 velas) y se te pregunta de nuevo: `vender` o `esperar`.
4. `esperar` → se respeta tu decisión y ese libro entra en cooldown hasta
   que la situación cambie.

### Anti-spam

- **Si ya compraste una cripto, no recibes más propuestas de esa cripto
  hasta que vendas.**
- Una sola propuesta de compra pendiente a la vez (expira a los 10 min).
- Cooldown por libro tras comprar/cancelar (30 min) y tras "esperar" (20 min).
- Una confirmación de venta pendiente por libro; sin re-preguntas por cruce
  de umbral repetido.

### Comandos manuales (siguen disponibles)

`compré <book> <qty> <precio>`, `vendí <book> <precio>`, `posiciones`,
`historial [book] [n]`, `estado`, `ayuda`.

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
│   │   ├── bitso/              # auth HMAC v3 (nonce v2) + REST + WebSocket
│   │   └── gbm/                # (Fase 2)
│   ├── risk/                   # (Fase 4)
│   └── services/               # orquestación entre capas
│       ├── monitor.py           # loop de escaneo + delegación al asistente
│       ├── trading_assistant.py # conversación de compra/venta con confirmación
│       ├── trader.py            # órdenes de mercado reales + fills (Bitso)
│       ├── wait_analyzer.py     # estadística "¿qué pasaría si espero?"
│       └── telegram_bot.py      # transporte Telegram (comandos + respuestas)
└── tests/
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

1. Depositar MXN en Bitso para que el asistente pueda comprar (con el saldo
   actual la compra real responderá "Saldo insuficiente").
2. Correr `python -m trading_system.main scan --interactive` y confirmar el
   flujo completo por Telegram.
3. Conectar el motor de riesgo (Fase 4) y la inferencia XGBoost (Fase 5) al
   flujo live del asistente.
