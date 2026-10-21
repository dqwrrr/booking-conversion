"""Проверка качества исходной части и порций по требованиям Таблицы 1.4.

Справочники (маршруты, длительность перелёта по маршруту, доля
положительных) строятся один раз по исходной части и сохраняются в
reports/quality/reference.json; порции сверяются с ними.
"""
import json
import sys
import time
from pathlib import Path

import pandas as pd
import pandera.pandas as pa
import yaml

sys.path.insert(0, str(Path(__file__).resolve().parents[2]))
from src.data.schema import FEATURES, labels, rows, structure  # noqa: E402

OUT = Path("reports/quality")


def build_reference(df: pd.DataFrame) -> dict:
    return {
        "routes": sorted(df["route"].unique().tolist()),
        "route_duration": df.groupby("route")["flight_duration"].first().to_dict(),
        "positive_share": float(df["booking_complete"].mean()),
    }


def check(df: pd.DataFrame, ref: dict, q: dict, lab: pd.DataFrame | None = None) -> dict:
    t0 = time.perf_counter()
    rep = {"rows_in": int(len(df)), "critical": [], "warnings": []}
    missing = [c for c in ["booking_id"] + FEATURES if c not in df.columns]
    if missing:
        rep["critical"].append(f"нет полей: {missing}")
        return rep
    try:
        structure.validate(df)
    except pa.errors.SchemaError as e:
        rep["critical"].append(f"ключ booking_id: {e}")
        return rep
    try:
        rows.validate(df, lazy=True)
        bad_idx = set()
    except pa.errors.SchemaErrors as e:
        fc = e.failure_cases
        bad_idx = set(fc["index"].dropna().astype(int))
    rep["rows_rejected"] = len(bad_idx)
    ok = df.drop(index=list(bad_idx))

    dup = ok.duplicated(subset=FEATURES + (["booking_complete"] if "booking_complete" in ok else []))
    rep["duplicates_removed"] = int(dup.sum())
    if dup.mean() > 0.01:
        rep["warnings"].append("доля дубликатов выше 1 %")
    ok = ok[~dup]

    known = ok["route"].map(ref["route_duration"])
    rep["duration_mismatch"] = int(((known.notna()) & (known != ok["flight_duration"])).sum())
    if rep["duration_mismatch"]:
        rep["warnings"].append("длительность перелёта расходится со справочной")
    new_share = float((~ok["route"].isin(ref["routes"])).mean())
    rep["new_route_share"] = round(new_share, 4)
    if new_share > q["new_route_share_max"]:
        rep["warnings"].append("доля новых маршрутов выше порога")
    rep["not_set_origin"] = int((ok["booking_origin"] == "(not set)").sum())
    if rep["not_set_origin"] > 0.01 * len(ok):
        rep["warnings"].append("доля booking_origin = (not set) выше 1 %")
    rep["stay_over_365"] = int((ok["length_of_stay"] > q["length_of_stay_flag"]).sum())
    rep["stay_7_16"] = int(ok["length_of_stay"].between(7, 16).sum())
    if rep["stay_7_16"]:
        rep["warnings"].append("появились значения length_of_stay 7-16")
    rep["oneway_with_stay"] = int(((ok["trip_type"] == "OneWay") & (ok["length_of_stay"] > 0)).sum())
    rep["lead_over_365"] = int((ok["purchase_lead"] > 365).sum())

    if lab is not None:
        labels.validate(lab)
        rep["labels_missing"] = int((~ok["booking_id"].isin(lab["booking_id"])).sum())
        pos = float(lab["booking_complete"].mean())
        rep["positive_share"] = round(pos, 4)
        dev = abs(pos - ref["positive_share"]) / ref["positive_share"]
        if dev > q["positive_share_tolerance"]:
            rep["warnings"].append("доля положительных вне допуска")
    rep["rows_out"] = int(len(ok))
    rep["seconds"] = round(time.perf_counter() - t0, 3)
    return rep


def main() -> None:
    q = yaml.safe_load(open("params.yaml", encoding="utf-8"))["quality"]
    OUT.mkdir(parents=True, exist_ok=True)
    init = pd.read_csv("data/interim/initial.csv")
    ref = build_reference(init.drop_duplicates(subset=FEATURES + ["booking_complete"]))
    json.dump(ref, open(OUT / "reference.json", "w", encoding="utf-8"), ensure_ascii=False, indent=1)
    reports = {"initial": check(init, ref, q)}
    for f in sorted(Path("data/incoming").glob("batch_*.csv")):
        k = f.stem.split("_")[1]
        lab = pd.read_csv(f"data/incoming/labels_{k}.csv")
        reports[f.stem] = check(pd.read_csv(f), ref, q, lab)
    # время проверки меняется от запуска к запуску, поэтому хранится отдельно
    # и не входит в отслеживаемый DVC отчёт
    timing = {k: v.pop("seconds", None) for k, v in reports.items()}
    json.dump(reports, open(OUT / "quality_report.json", "w", encoding="utf-8"), ensure_ascii=False, indent=1)
    json.dump(timing, open(OUT / "timing.json", "w", encoding="utf-8"), indent=1)
    print("время проверки, с:", timing)
    for k, v in reports.items():
        print(k, v)


if __name__ == "__main__":
    main()
