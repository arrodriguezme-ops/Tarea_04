"""Baseline 1: TF-IDF (word 1-2 grams) + Ridge, alpha tuned on validation MAE.

Numbers are kept by default: years and dates are the evidence for experience, so removing them
would deliberately blind the model. `--no-digits` removes them for the Question 3 comparison.
"""
import argparse
import re
import time

import joblib
import numpy as np
import pandas as pd
from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.linear_model import Ridge

from utils import CAP, CLEAN_TEXT_COL, METRICS, MODELS, REF_TEXT_COL, clip_years, save_json, split_with_text

ALPHAS = [0.1, 0.3, 1, 3, 10, 30]


def strip_digits(t):
    return re.sub(r"\d+", " ", t)


def build(keep_digits=True):
    return TfidfVectorizer(
        ngram_range=(1, 2), min_df=3, max_df=0.95, sublinear_tf=True, lowercase=True,
        token_pattern=r"(?u)\b\w+\b" if keep_digits else r"(?u)\b[^\W\d_]+\b",
    )


def path(tag):
    return MODELS / "baselines" / f"tfidf{tag}.joblib"


def predict(texts, tag=""):
    vec, model = joblib.load(path(tag))
    return clip_years(model.predict(vec.transform(texts)))


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--text-col", default=CLEAN_TEXT_COL)
    ap.add_argument("--no-digits", action="store_true")
    ap.add_argument("--tag", default="", help="suffix for sensitivity runs, e.g. __nodigits")
    a = ap.parse_args()
    tag = a.tag or ("__nodigits" if a.no_digits else "")
    tr, va, te = (split_with_text(s) for s in ("train", "val", "test"))
    t0 = time.time()
    vec = build(not a.no_digits)
    Xtr = vec.fit_transform(tr[a.text_col])
    Xva = vec.transform(va[a.text_col])
    best = min(
        ((np.abs(clip_years(Ridge(alpha=al).fit(Xtr, tr["exp_year"]).predict(Xva)) - va["exp_year"]).mean(), al)
         for al in ALPHAS))
    model = Ridge(alpha=best[1]).fit(Xtr, tr["exp_year"])
    train_s = time.time() - t0
    joblib.dump((vec, model), path(tag))
    t0 = time.time()
    p_ocr = clip_years(model.predict(vec.transform(te[a.text_col])))
    inf = (time.time() - t0) / len(te)
    p_ref = clip_years(model.predict(vec.transform(te[REF_TEXT_COL])))
    pd.DataFrame({"id": te["id"], "pred_ocr": p_ocr, "pred_ref": p_ref}).to_csv(
        METRICS / f"preds_tfidf{tag}.csv", index=False)
    save_json({"train_s": train_s, "infer_s_per_cv": inf, "alpha": best[1], "val_mae": best[0]},
              METRICS / f"time_tfidf{tag}.json")
    print(f"alpha={best[1]} val MAE={best[0]:.3f}")


if __name__ == "__main__":
    main()
