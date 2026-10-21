import subprocess
import sys
from pathlib import Path

import pandas as pd
import yaml

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))
from src.data.validate import build_reference, check  # noqa: E402

Q = yaml.safe_load(open("params.yaml", encoding="utf-8"))["quality"]


def test_split_is_deterministic_and_disjoint(tmp_path):
    before = pd.read_csv("data/incoming/batch_01.csv")["booking_id"].tolist()
    subprocess.run([sys.executable, "src/data/split_batches.py"], check=True)
    after = pd.read_csv("data/incoming/batch_01.csv")["booking_id"].tolist()
    assert before == after
    ids = set(pd.read_csv("data/interim/initial.csv")["booking_id"])
    total = len(ids)
    for k in range(1, 5):
        b = set(pd.read_csv(f"data/incoming/batch_{k:02d}.csv")["booking_id"])
        assert not ids & b
        ids |= b
        total += len(b)
    assert len(ids) == total == 50000


def test_batch_has_no_target():
    b = pd.read_csv("data/incoming/batch_01.csv")
    assert "booking_complete" not in b.columns


def test_bad_rows_and_missing_field():
    init = pd.read_csv("data/interim/initial.csv")
    ref = build_reference(init)
    b = pd.read_csv("data/incoming/batch_01.csv").head(100).copy()
    b.loc[0, "purchase_lead"] = 1200
    b.loc[1, "sales_channel"] = "Phone"
    rep = check(b, ref, Q)
    assert rep["rows_rejected"] == 2 and not rep["critical"]
    rep = check(b.drop(columns=["route"]), ref, Q)
    assert rep["critical"]
