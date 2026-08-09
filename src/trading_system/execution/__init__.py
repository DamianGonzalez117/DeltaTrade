"""Capa de ejecución y orquestación del sistema de trading.

Exporta:
  - ``OrderExecutor``: ejecuta órdenes aprobadas via ExchangeConnector.
  - ``ExecutionManager``: orquesta inferencia → riesgo → ejecución.
  - ``PortfolioProvider``: interfaz para obtener estado del portafolio.
"""
from trading_system.execution.executor import OrderExecutor
from trading_system.execution.manager import ExecutionManager, PortfolioProvider

__all__ = ["ExecutionManager", "OrderExecutor", "PortfolioProvider"]