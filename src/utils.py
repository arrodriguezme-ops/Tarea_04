"""Seeds, paths and helpers shared by every stage."""
import json
import os
import random
from pathlib import Path

import numpy as np
import pandas as pd

ROOT = Path(__file__).resolve().parents[1]
RAW = ROOT / "data" / "raw"
IMAGES = RAW / "images"
OCR_DIR = ROOT / "data" / "ocr"
PROCESSED = ROOT / "data" / "processed"
SPLITS = ROOT / "data" / "splits"
MODELS = ROOT / "models"
OUT = ROOT / "output"
FIGS = OUT / "figures"
TABLES = OUT / "tables"
METRICS = OUT / "metrics"

SEED = 42
CAP = 20  # years; values above become "20 or more". Same cap for every model.
BAND_EDGES = [0, 3, 6, 11, CAP + 1]  # 0-2, 3-5, 6-10, 11-20
BAND_LABELS = ["0-2", "3-5", "6-10", "11-20"]
JOBS = {
    "registered_nurses": "nurse",
    "elementary_middle_school_teachers": "teacher",
    "software_developers": "software",
    "accountants_auditors": "account",
    "construction_laborers": "construction",
}
CLEAN_TEXT_COL = "text_clean"
REF_TEXT_COL = "text_ref"


def set_seed(seed: int = SEED) -> None:
    random.seed(seed)
    np.random.seed(seed)
    os.environ["PYTHONHASHSEED"] = str(seed)
    try:
        import torch

        torch.manual_seed(seed)
        torch.cuda.manual_seed_all(seed)
    except ImportError:
        pass


def clip_years(y):
    return np.clip(y, 0, CAP)


def to_band(y):
    """Years -> band label (accepts scalars or arrays; floats are floored)."""
    y = np.floor(clip_years(np.asarray(y, dtype=float)))
    return pd.cut(y, bins=BAND_EDGES, labels=BAND_LABELS, right=False).astype(str)


def load_split(name: str) -> pd.DataFrame:
    return pd.read_csv(SPLITS / f"{name}.csv", dtype={"id": str})


def load_processed() -> pd.DataFrame:
    return pd.read_csv(PROCESSED / "clean.csv", dtype={"id": str}).fillna("")


def split_with_text(name: str) -> pd.DataFrame:
    """Split IDs joined with their processed texts and target."""
    return load_split(name)[["id"]].merge(load_processed(), on="id", how="left")


def save_json(obj, path) -> None:
    Path(path).parent.mkdir(parents=True, exist_ok=True)
    with open(path, "w", encoding="utf-8") as f:
        json.dump(obj, f, indent=2, ensure_ascii=False)


def load_json(path):
    with open(path, encoding="utf-8") as f:
        return json.load(f)
