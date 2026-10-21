"""Разбиение исходной выгрузки на исходную часть и порции «новых» данных.

Даты создания брони в наборе нет, поэтому порции формируются случайно
с фиксированным начальным значением генератора (params.yaml: split.seed).
Каждой записи присваивается суррогатный ключ booking_id по номеру строки
в исходном файле. Метка booking_complete в порции не включается: она
выдаётся отдельным файлом labels_XX.csv, как в эксплуатации, где исход
брони становится известен только после истечения таймлимита.
"""
from pathlib import Path

import numpy as np
import pandas as pd
import yaml

TARGET = "booking_complete"


def main() -> None:
    p = yaml.safe_load(open("params.yaml", encoding="utf-8"))["split"]
    raw = pd.read_csv(p["source"], encoding=p["encoding"])
    raw.insert(0, "booking_id", [f"B{i:06d}" for i in range(1, len(raw) + 1)])

    rng = np.random.default_rng(p["seed"])
    order = rng.permutation(len(raw))
    n_init = int(round(len(raw) * p["initial_share"]))
    init_idx, rest_idx = order[:n_init], order[n_init:]

    Path("data/interim").mkdir(parents=True, exist_ok=True)
    Path("data/incoming").mkdir(parents=True, exist_ok=True)
    raw.iloc[np.sort(init_idx)].to_csv("data/interim/initial.csv", index=False)

    for k, part in enumerate(np.array_split(rest_idx, p["n_batches"]), start=1):
        b = raw.iloc[np.sort(part)]
        b.drop(columns=[TARGET]).to_csv(f"data/incoming/batch_{k:02d}.csv", index=False)
        b[["booking_id", TARGET]].to_csv(f"data/incoming/labels_{k:02d}.csv", index=False)
        print(f"batch_{k:02d}: {len(b)} записей")
    print(f"initial: {n_init} записей")


if __name__ == "__main__":
    main()
