"""Metrics, model comparison, bootstrap gains, OCR gap, fairness, probe and error tables.

python src/evaluation.py          -> writes everything in output/tables and output/metrics
"""
import re

import numpy as np
import pandas as pd
from sklearn.metrics import confusion_matrix, f1_score, r2_score

from utils import BAND_LABELS, CLEAN_TEXT_COL, METRICS, RAW, TABLES, clip_years, load_json, load_processed, load_split, to_band

MODELS = {"median": "Median", "tfidf": "TF-IDF + Ridge", "embeddings": "Embeddings + Ridge", "bert": "Fine-tuned BERT"}
N_BOOT = 2000


def compute_metrics(y, p):
    y, p = np.asarray(y, float), clip_years(np.asarray(p, float))
    err = np.abs(p - y)
    by, bp = to_band(y), to_band(p)
    return {
        "MAE": err.mean(), "RMSE": float(np.sqrt(((p - y) ** 2).mean())),
        "Acc±1": (err <= 1).mean(), "Acc±2": (err <= 2).mean(), "Acc±3": (err <= 3).mean(),
        "Band accuracy": (by == bp).mean(),
        "Band macro-F1": f1_score(by, bp, labels=BAND_LABELS, average="macro", zero_division=0),
        "R2": r2_score(y, p),
    }


def load_test():
    """Test split with metadata (sex, race) and all prediction files merged on id."""
    te = load_split("test")[["id", "job", "exp_year", "band"]]
    meta = pd.read_csv(RAW / "metadata.csv", dtype={"id": str})[["id", "sex", "race"]]
    te = te.merge(meta, on="id", how="left")
    for m in MODELS:
        pr = pd.read_csv(METRICS / f"preds_{m}.csv", dtype={"id": str})
        te = te.merge(pr.rename(columns={"pred_ocr": f"{m}", "pred_ref": f"{m}_ref"}), on="id")
    return te


def bootstrap_gain(y, p_base, p_bert, n=N_BOOT, seed=42):
    """Paired bootstrap of MAE(baseline) - MAE(BERT); positive = BERT better."""
    rng = np.random.default_rng(seed)
    d = np.abs(p_base - y) - np.abs(p_bert - y)
    idx = rng.integers(0, len(y), (n, len(y)))
    g = d[idx].mean(1)
    return d.mean(), np.percentile(g, 2.5), np.percentile(g, 97.5)


def shift_years(text, k):
    return re.sub(r"\b(19|20)(\d{2})\b", lambda m: str(int(m.group(0)) + k), text)


def probe(n=20, k=5, seed=42):
    """Shift every year in 20 test CVs by +k and see how each model reacts.

    A recruiter would say experience is unchanged for fully-dated careers (all durations stay the
    same). A model that reads vocabulary only changes little; one that reads dates relative to a
    fixed 'now' sees a career that ends in the future and moves.
    """
    import baseline_embeddings, baseline_median, baseline_tfidf
    from train_bert import load_predictor

    df = load_processed().merge(load_split("test")[["id"]], on="id").sample(n, random_state=seed)
    shifted = df[CLEAN_TEXT_COL].map(lambda t: shift_years(t, k))
    preds = {"median": baseline_median.predict, "tfidf": baseline_tfidf.predict,
             "embeddings": baseline_embeddings.predict, "bert": load_predictor()}
    rows = []
    for m, f in preds.items():
        a, b = f(list(df[CLEAN_TEXT_COL])), f(list(shifted))
        rows.append({"model": MODELS[m], "mean_change": float((b - a).mean()), "mean_abs_change": float(np.abs(b - a).mean())})
    return pd.DataFrame(rows)


def main():
    te = load_test()
    y = te["exp_year"].values
    tm = {m: load_json(METRICS / f"time_{m}.json") for m in MODELS}

    rows = []
    for m, name in MODELS.items():
        r = {"Model": name, **compute_metrics(y, te[m])}
        r["Training time (s)"] = tm[m]["train_s"]
        r["Inference time per CV (s)"] = tm[m]["infer_s_per_cv"]
        rows.append(r)
    comp = pd.DataFrame(rows).round(4)
    comp.to_csv(TABLES / "model_comparison.csv", index=False)
    print(comp.to_string(index=False))

    # BERT gain with CI
    g = []
    for m in ["median", "tfidf", "embeddings"]:
        mean, lo, hi = bootstrap_gain(y, te[m].values, te["bert"].values)
        g.append({"Baseline": MODELS[m], "MAE gain (years)": mean, "CI95 low": lo, "CI95 high": hi,
                  "Significant": bool(lo > 0 or hi < 0)})
    pd.DataFrame(g).round(4).to_csv(TABLES / "bert_gain.csv", index=False)

    # MAE by band and by occupation
    for col, fn in [("band", "mae_by_band"), ("job", "mae_by_job")]:
        t = pd.DataFrame({MODELS[m]: te.groupby(col).apply(lambda d, m=m: np.abs(d[m] - d["exp_year"]).mean(), include_groups=False)
                          for m in MODELS})
        t.round(4).to_csv(TABLES / f"{fn}.csv")

    # Confusion matrices and bias towards the mean
    for m in MODELS:
        cm = confusion_matrix(to_band(y), to_band(te[m]), labels=BAND_LABELS)
        pd.DataFrame(cm, index=BAND_LABELS, columns=BAND_LABELS).to_csv(TABLES / f"confusion_{m}.csv")
    pd.DataFrame({MODELS[m]: te.groupby("band").apply(lambda d, m=m: (d[m] - d["exp_year"]).mean(), include_groups=False)
                  for m in MODELS}).round(4).to_csv(TABLES / "bias_by_band.csv")

    # Q2: OCR text vs reference text
    q2 = [{"Model": name, "MAE OCR text": np.abs(te[m] - y).mean(), "MAE reference text": np.abs(te[f"{m}_ref"] - y).mean()}
          for m, name in MODELS.items()]
    q2 = pd.DataFrame(q2)
    q2["Gap (OCR - reference)"] = q2["MAE OCR text"] - q2["MAE reference text"]
    q2.round(4).to_csv(TABLES / "ocr_gap.csv", index=False)

    # Q4: fairness (sex / race are NOT model inputs)
    fair = []
    for col in ["sex", "race"]:
        for grp, d in te.groupby(col):
            e = d["bert"] - d["exp_year"]
            fair.append({"attribute": col, "group": grp, "n": len(d), "MAE": np.abs(e).mean(), "mean_error (pred-true)": e.mean()})
    pd.DataFrame(fair).round(4).to_csv(TABLES / "fairness_bert.csv", index=False)

    # 10 largest BERT errors, to be categorised by hand (cause column)
    te["abs_err"] = np.abs(te["bert"] - te["exp_year"])
    top = te.nlargest(10, "abs_err")[["id", "job", "exp_year", "bert", "abs_err"]].copy()
    top["cause"] = ""
    top.round(3).to_csv(TABLES / "top10_bert_errors.csv", index=False)

    # Q3: every preds_<model>__<tag>.csv found
    sens = []
    for f in sorted(METRICS.glob("preds_*__*.csv")):
        m, tag = f.stem[len("preds_"):].split("__", 1)
        pr = pd.read_csv(f, dtype={"id": str}).merge(te[["id", "exp_year"]], on="id")
        sens.append({"model": MODELS.get(m, m), "variant": tag, "MAE": np.abs(pr["pred_ocr"] - pr["exp_year"]).mean()})
    if sens:
        pd.DataFrame(sens).round(4).to_csv(TABLES / "sensitivity.csv", index=False)

    try:
        probe().round(4).to_csv(TABLES / "probe_shift_dates.csv", index=False)
    except Exception as e:  # models not trained yet
        print("probe skipped:", e)


if __name__ == "__main__":
    main()
