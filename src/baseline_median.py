"""Baseline 0: always predict the training-set median."""
import time

import joblib
import numpy as np
import pandas as pd

from utils import METRICS, MODELS, REF_TEXT_COL, CLEAN_TEXT_COL, save_json, split_with_text

NAME = "median"
PATH = MODELS / "baselines" / "median.joblib"


def predict(texts):
    return np.full(len(texts), joblib.load(PATH))


def main():
    t0 = time.time()
    train = split_with_text("train")
    median = float(train["exp_year"].median())
    joblib.dump(median, PATH)
    train_s = time.time() - t0
    test = split_with_text("test")
    t0 = time.time()
    p = predict(test[CLEAN_TEXT_COL])
    inf = (time.time() - t0) / len(test)
    pd.DataFrame({"id": test["id"], "pred_ocr": p, "pred_ref": predict(test[REF_TEXT_COL])}).to_csv(
        METRICS / f"preds_{NAME}.csv", index=False)
    save_json({"train_s": train_s, "infer_s_per_cv": inf, "median": median}, METRICS / f"time_{NAME}.json")
    print("median =", median)


if __name__ == "__main__":
    main()
