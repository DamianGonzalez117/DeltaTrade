"""TDD — Tests for BMV market calendar (RED phase).

El BMV (Bolsa Mexicana de Valores) opera:
  - Lunes a Viernes (mostrando como 0-4 en Python's weekday())
  - 8:00 AM a 2:30 PM (hora de CDMX, UTC-6 invierno / UTC-5 verano)
  - Cerrado en días festivos bancarios mexicanos

El calendario debe responder ``is_market_open(dt)`` con un bool,
independientemente de la timezone del datetime recibido.
"""
from datetime import UTC, datetime
from zoneinfo import ZoneInfo

from trading_system.connectors.gbm.calendar import BMVCalendar

CDMX = ZoneInfo("America/Mexico_City")


class TestMarketOpenRegularDays:
    def test_monday_morning_open(self):
        cal = BMVCalendar()
        dt = datetime(2026, 7, 27, 10, 0, tzinfo=CDMX)  # lunes 10:00
        assert cal.is_market_open(dt) is True

    def test_tuesday_afternoon_open(self):
        cal = BMVCalendar()
        dt = datetime(2026, 7, 28, 13, 0, tzinfo=CDMX)  # martes 13:00
        assert cal.is_market_open(dt) is True

    def test_wednesday_morning_open(self):
        cal = BMVCalendar()
        dt = datetime(2026, 7, 29, 9, 30, tzinfo=CDMX)  # miércoles 9:30
        assert cal.is_market_open(dt) is True


class TestMarketClosedWeekend:
    def test_saturday_closed(self):
        cal = BMVCalendar()
        dt = datetime(2026, 8, 1, 10, 0, tzinfo=CDMX)  # sábado 10:00
        assert cal.is_market_open(dt) is False

    def test_sunday_closed(self):
        cal = BMVCalendar()
        dt = datetime(2026, 8, 2, 10, 0, tzinfo=CDMX)  # domingo 10:00
        assert cal.is_market_open(dt) is False

    def test_friday_after_close_closed(self):
        cal = BMVCalendar()
        dt = datetime(2026, 7, 31, 14, 31, tzinfo=CDMX)  # viernes 14:31
        assert cal.is_market_open(dt) is False

    def test_monday_before_open_closed(self):
        cal = BMVCalendar()
        dt = datetime(2026, 7, 27, 7, 59, tzinfo=CDMX)  # lunes 7:59
        assert cal.is_market_open(dt) is False

    def test_exactly_at_open(self):
        cal = BMVCalendar()
        dt = datetime(2026, 7, 27, 8, 0, tzinfo=CDMX)  # lunes 8:00
        assert cal.is_market_open(dt) is True

    def test_exactly_at_close(self):
        cal = BMVCalendar()
        dt = datetime(2026, 7, 27, 14, 30, tzinfo=CDMX)  # lunes 14:30
        assert cal.is_market_open(dt) is False


class TestMarketClosedHolidays:
    def test_new_year_day_closed(self):
        cal = BMVCalendar()
        dt = datetime(2026, 1, 1, 10, 0, tzinfo=CDMX)  # jueves 1 enero
        assert cal.is_market_open(dt) is False

    def test_constitution_day_2026(self):
        cal = BMVCalendar()
        dt = datetime(2026, 2, 2, 10, 0, tzinfo=CDMX)  # lunes 2 feb 2026
        assert cal.is_market_open(dt) is False

    def test_independence_day_2026(self):
        cal = BMVCalendar()
        dt = datetime(2026, 9, 16, 10, 0, tzinfo=CDMX)  # miércoles 16 sep
        assert cal.is_market_open(dt) is False

    def test_christmas_2026(self):
        cal = BMVCalendar()
        dt = datetime(2026, 12, 25, 10, 0, tzinfo=CDMX)  # viernes 25 dic
        assert cal.is_market_open(dt) is False

    def test_labour_day_2026(self):
        cal = BMVCalendar()
        dt = datetime(2026, 5, 1, 10, 0, tzinfo=CDMX)  # viernes 1 mayo
        assert cal.is_market_open(dt) is False


class TestTimezoneHandling:
    def test_utc_datetime_converted_to_cdmx(self):
        cal = BMVCalendar()
        dt = datetime(2026, 7, 27, 16, 0, tzinfo=UTC)  # 16:00 UTC = 10:00 CDMX (verano)
        assert cal.is_market_open(dt) is True

    def test_utc_datetime_outside_hours(self):
        cal = BMVCalendar()
        dt = datetime(2026, 7, 27, 21, 0, tzinfo=UTC)  # 21:00 UTC = 15:00 CDMX
        assert cal.is_market_open(dt) is False

    def test_naive_datetime_treated_as_cdmx(self):
        cal = BMVCalendar()
        dt = datetime(2026, 7, 27, 10, 0)  # noqa: DTZ001 — intentionally naive to test CDMX fallback
        assert cal.is_market_open(dt) is True


class TestNextOpenSession:
    def test_next_open_from_weekend_returns_monday(self):
        cal = BMVCalendar()
        dt = datetime(2026, 8, 1, 10, 0, tzinfo=CDMX)  # sábado
        next_open = cal.next_open_session(dt)
        assert next_open.weekday() == 0  # lunes
        assert next_open.hour == 8
        assert next_open.minute == 0

    def test_next_open_from_afternoon_same_day_returns_next_day(self):
        cal = BMVCalendar()
        dt = datetime(2026, 7, 27, 15, 0, tzinfo=CDMX)  # lunes 15:00
        next_open = cal.next_open_session(dt)
        assert next_open.day == 28
        assert next_open.hour == 8

    def test_next_open_from_holiday_skips_holiday(self):
        cal = BMVCalendar()
        dt = datetime(2026, 9, 16, 10, 0, tzinfo=CDMX)  # miércoles 16 sep (Independence)
        next_open = cal.next_open_session(dt)
        assert next_open.day == 17
        assert next_open.hour == 8


class TestIsTradingDay:
    def test_regular_monday_is_trading_day(self):
        cal = BMVCalendar()
        dt = datetime(2026, 7, 27, 10, 0, tzinfo=CDMX)  # lunes
        assert cal.is_trading_day(dt.date()) is True

    def test_saturday_is_not_trading_day(self):
        cal = BMVCalendar()
        dt = datetime(2026, 8, 1, 10, 0, tzinfo=CDMX)  # sábado
        assert cal.is_trading_day(dt.date()) is False

    def test_holiday_is_not_trading_day(self):
        cal = BMVCalendar()
        dt = datetime(2026, 12, 25, 10, 0, tzinfo=CDMX)  # christmas
        assert cal.is_trading_day(dt.date()) is False