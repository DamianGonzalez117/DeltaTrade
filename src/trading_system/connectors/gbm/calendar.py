"""
Calendario del BMV (Bolsa Mexicana de Valores) / SIC.

Maneja:
  - Horario regular: Lun-Vie 8:00–14:30 (hora de CDMX)
  - Fines de semana: cerrado
  - Días festivos mexicanos: fecha fija y fecha móvil

SRP: solo conoce el calendario. No toca HTTP ni configuración.
OCP: nuevos festivos se añaden sin modificar ``is_market_open``.
"""
from datetime import date, datetime, timedelta
from zoneinfo import ZoneInfo

CDMX_TZ = ZoneInfo("America/Mexico_City")
_MARKET_OPEN_HOUR = 8
_MARKET_CLOSE_HOUR = 14
_MARKET_CLOSE_MINUTE = 30


def _fixed_holidays(year: int) -> set[date]:
    """Festivos con fecha fija del calendario bancario mexicano."""
    holidays = {
        date(year, 1, 1),    # Año Nuevo
        date(year, 5, 1),   # Día del Trabajo
        date(year, 9, 16),  # Independencia
        date(year, 12, 25), # Navidad
    }

    # Festivos móviles: primer lunes del mes (Ley de reposición de festivos)
    def first_monday(month: int) -> date:
        d = date(year, month, 1)
        offset = (7 - d.weekday()) % 7
        return d + timedelta(days=offset)

    holidays.add(first_monday(2))   # Constitución (primer lunes de febrero)
    holidays.add(first_monday(3))   # Benito Juárez (primer lunes de marzo)

    # Cuarto lunes de noviembre: Revolución Mexicana
    nov1 = date(year, 11, 1)
    first_monday_nov = nov1 + timedelta(days=(7 - nov1.weekday()) % 7)
    holidays.add(first_monday_nov + timedelta(weeks=3))

    # Cuando un festivo fijo cae en fin de semana, se reposiciona al lunes siguiente
    extra = set()
    for h in holidays:
        if h.weekday() == 5:  # sábado
            extra.add(h + timedelta(days=2))
        elif h.weekday() == 6:  # domingo
            extra.add(h + timedelta(days=1))
    holidays |= extra

    return holidays


class BMVCalendar:
    """Determina si el BMV está abierto en un momento dado."""

    def __init__(self) -> None:
        self._holiday_cache: dict[int, set[date]] = {}

    def _holidays_for(self, year: int) -> set[date]:
        if year not in self._holiday_cache:
            self._holiday_cache[year] = _fixed_holidays(year)
        return self._holiday_cache[year]

    def is_trading_day(self, d: date) -> bool:
        if d.weekday() >= 5:
            return False
        return d not in self._holidays_for(d.year)

    def is_market_open(self, dt: datetime) -> bool:
        local = self._to_cdmx(dt)
        if not self.is_trading_day(local.date()):
            return False
        market_open = local.replace(hour=_MARKET_OPEN_HOUR, minute=0, second=0, microsecond=0)
        market_close = local.replace(
            hour=_MARKET_CLOSE_HOUR, minute=_MARKET_CLOSE_MINUTE, second=0, microsecond=0
        )
        return market_open <= local < market_close

    def next_open_session(self, dt: datetime) -> datetime:
        local = self._to_cdmx(dt)
        candidate = local.replace(hour=_MARKET_OPEN_HOUR, minute=0, second=0, microsecond=0)
        if candidate <= local:
            candidate += timedelta(days=1)
        while not self.is_trading_day(candidate.date()):
            candidate += timedelta(days=1)
        return candidate

    @staticmethod
    def _to_cdmx(dt: datetime) -> datetime:
        if dt.tzinfo is None:
            return dt.replace(tzinfo=CDMX_TZ)
        return dt.astimezone(CDMX_TZ)