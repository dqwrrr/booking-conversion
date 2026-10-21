"""Требования к качеству данных (отчёт ПР2, Таблица 1.4) в виде проверок pandera.

CRITICAL — нарушение останавливает обработку порции целиком.
ROW      — запись исключается, число исключённых пишется в отчёт.
WARN     — порция обрабатывается, в отчёт пишется предупреждение.
"""
import pandera.pandas as pa
from pandera import Check

CHANNELS = ["Internet", "Mobile"]
TRIPS = ["RoundTrip", "OneWay", "CircleTrip"]
DAYS = ["Mon", "Tue", "Wed", "Thu", "Fri", "Sat", "Sun"]
FEATURES = ["num_passengers", "sales_channel", "trip_type", "purchase_lead",
            "length_of_stay", "flight_hour", "flight_day", "route",
            "booking_origin", "wants_extra_baggage", "wants_preferred_seat",
            "wants_in_flight_meals", "flight_duration"]

# Т1, Т2: структура порции и ключ — критические
structure = pa.DataFrameSchema(
    {"booking_id": pa.Column(str, unique=True, nullable=False)},
    strict=False)

# Т3–Т6: построчные правила — нарушившая запись исключается
rows = pa.DataFrameSchema({
    "num_passengers": pa.Column(int, Check.in_range(1, 9)),
    "sales_channel": pa.Column(str, Check.isin(CHANNELS)),
    "trip_type": pa.Column(str, Check.isin(TRIPS)),
    "purchase_lead": pa.Column(int, Check.in_range(0, 900)),
    "length_of_stay": pa.Column(int, Check.ge(0)),
    "flight_hour": pa.Column(int, Check.in_range(0, 23)),
    "flight_day": pa.Column(str, Check.isin(DAYS)),
    "route": pa.Column(str, Check.str_matches(r"^[A-Z]{6}$")),
    "booking_origin": pa.Column(str, nullable=False),
    "wants_extra_baggage": pa.Column(int, Check.isin([0, 1])),
    "wants_preferred_seat": pa.Column(int, Check.isin([0, 1])),
    "wants_in_flight_meals": pa.Column(int, Check.isin([0, 1])),
    "flight_duration": pa.Column(float, Check.in_range(1.0, 20.0)),
}, strict=False)

labels = pa.DataFrameSchema({
    "booking_id": pa.Column(str, unique=True, nullable=False),
    "booking_complete": pa.Column(int, Check.isin([0, 1])),
})
