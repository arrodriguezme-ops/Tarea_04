"""Baseline 2: frozen sentence encoder + Ridge.

all-MiniLM-L6-v2 truncates at 256 word pieces, so a CV is split into overlapping chunks of
~120 words (stride 90), each chunk is encoded and the chunk vectors are averaged. The encoder
is never fine-tuned.
"""
import argparse
import time

import joblib
import numpy as np
import pandas as pd
from sklearn.linear_model import Ridge

from utils import CLEAN_TEXT_COL, METRICS, MODELS, REF_TEXT_COL, clip_years, save_json, split_with_text

ENCODER = "sentence-transformers/all-MiniLM-L6-v2"
ALPHAS = [0.1, 0.3, 1, 3, 10, 30, 100]
_ST = None


def chunks(text, size=120, stride=90):
    w = text.split()
    if len(w) <= size:
        return [text or " "]
    return [" ".join(w[i:i + size]) for i in range(0, len(w) - size + stride, stride)]


def encode(texts):
    global _ST
    if _ST is None:
        from sentence_transformers import SentenceTransformer

        _ST = SentenceTransformer(ENCODER)
    parts = [chunks(t) for t in texts]
    flat = [c for p in parts for c in p]
    emb = _ST.encode(flat, batch_size=64, normalize_embeddings=True, show_progress_bar=False)
    out, k = [], 0
    for p in parts:
        out.append(emb[k:k + len(p)].mean(0))
        k += len(p)
    return np.vstack(out)


def predict(texts, tag=""):
    return clip_years(joblib.load(MODELS / "baselines" / f"embeddings{tag}.joblib").predict(encode(texts)))


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--text-col", default=CLEAN_TEXT_COL)
    ap.add_argument("--tag", default="")
    a = ap.parse_args()
    tr, va, te = (split_with_text(s) for s in ("train", "val", "test"))
    t0 = time.time()
    Xtr, Xva = encode(tr[a.text_col]), encode(va[a.text_col])
    best = min(((np.abs(clip_years(Ridge(alpha=al).fit(Xtr, tr["exp_year"]).predict(Xva)) - va["exp_year"]).mean(), al)
                for al in ALPHAS))
    model = Ridge(alpha=best[1]).fit(Xtr, tr["exp_year"])
    train_s = time.time() - t0
    joblib.dump(model, MODELS / "baselines" / f"embeddings{a.tag}.joblib")
    t0 = time.time()
    p_ocr = clip_years(model.predict(encode(te[a.text_col])))
    inf = (time.time() - t0) / len(te)
    p_ref = clip_years(model.predict(encode(te[REF_TEXT_COL])))
    pd.DataFrame({"id": te["id"], "pred_ocr": p_ocr, "pred_ref": p_ref}).to_csv(
        METRICS / f"preds_embeddings{a.tag}.csv", index=False)
    save_json({"train_s": train_s, "infer_s_per_cv": inf, "alpha": best[1], "val_mae": best[0]},
              METRICS / f"time_embeddings{a.tag}.json")
    print(f"alpha={best[1]} val MAE={best[0]:.3f}")


if __name__ == "__main__":
    main()
