# booking-conversion

Прогнозирование завершения бронирования авиабилета (дойдёт ли бронь до выписки).
Учебный проект по дисциплине «Технологии и инструментарий машинного обучения»,
Петров А. В., ИМБО-02-24.

## Запуск

```bash
uv sync                  # окружение по uv.lock (Python 3.11)
uv run dvc pull          # исходная выгрузка из хранилища DVC
uv run dvc repro         # split -> validate -> eda
uv run pytest -q
```

## Данные

- `data/raw/customer_booking.csv` — исходная выгрузка Customer Booking Behaviour,
  50 000 строк, latin-1; под DVC, файл не изменяется.
- `data/interim/initial.csv` — исходная часть, 60 % (30 000 записей).
- `data/incoming/batch_01..04.csv` — порции «новых» данных по 5 000 записей без
  целевой переменной; `labels_01..04.csv` — метки к ним, поступают позже порции.
- Разбиение случайное (даты брони в наборе нет), `params.yaml: split.seed = 42`.
