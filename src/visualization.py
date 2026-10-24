"""Figures saved to output/figures (also reused by the Streamlit app)."""
import matplotlib

matplotlib.use("Agg")
import matplotlib.pyplot as plt
import pandas as pd
import seaborn as sns

from evaluation import MODELS, load_test
from utils import BAND_LABELS, CAP, FIGS, TABLES, load_processed

sns.set_theme(style="whitegrid")


def fig_distribution():
    df = load_processed()
    fig, ax = plt.subplots(1, 2, figsize=(11, 4))
    sns.histplot(df["exp_year"], bins=CAP, ax=ax[0])
    ax[0].set(title=f"Years of experience (clipped at {CAP})", xlabel="years")
    sns.boxplot(data=df, y="job", x="exp_year", ax=ax[1])
    ax[1].set(title="By occupation", xlabel="years", ylabel="")
    fig.tight_layout()
    return fig


def fig_ocr_quality():
    q = pd.read_csv(TABLES / "ocr_quality_per_cv.csv")
    fig, ax = plt.subplots(1, 3, figsize=(13, 3.6))
    for a, c in zip(ax, ["cer", "wer", "years_correct"]):
        sns.histplot(q[c].dropna(), bins=30, ax=a)
        a.set_title(c.upper() if c != "years_correct" else "Years read correctly")
    fig.tight_layout()
    return fig


def fig_pred_vs_true(model):
    te = load_test()
    fig, ax = plt.subplots(figsize=(5, 5))
    ax.scatter(te["exp_year"], te[model], s=10, alpha=0.4)
    ax.plot([0, CAP], [0, CAP], "r--", label="45° line")
    ax.set(xlabel="True years", ylabel="Predicted years", title=MODELS[model], xlim=(0, CAP), ylim=(0, CAP))
    ax.legend()
    fig.tight_layout()
    return fig


def fig_mae_grouped(table):
    t = pd.read_csv(TABLES / f"{table}.csv", index_col=0)
    ax = t.plot.bar(figsize=(9, 4), rot=0 if table == "mae_by_band" else 20)
    ax.set_ylabel("MAE (years)")
    plt.tight_layout()
    return ax.figure


def fig_confusion(model):
    cm = pd.read_csv(TABLES / f"confusion_{model}.csv", index_col=0)
    fig, ax = plt.subplots(figsize=(4.5, 4))
    sns.heatmap(cm, annot=True, fmt="d", cmap="Blues", cbar=False, ax=ax)
    ax.set(xlabel="Predicted band", ylabel="True band", title=MODELS[model])
    fig.tight_layout()
    return fig


def main():
    FIGS.mkdir(parents=True, exist_ok=True)
    fig_distribution().savefig(FIGS / "distribution.png", dpi=150)
    try:
        fig_ocr_quality().savefig(FIGS / "ocr_quality.png", dpi=150)
    except FileNotFoundError:
        pass
    for m in MODELS:
        fig_pred_vs_true(m).savefig(FIGS / f"pred_vs_true_{m}.png", dpi=150)
        fig_confusion(m).savefig(FIGS / f"confusion_{m}.png", dpi=150)
    for t in ["mae_by_band", "mae_by_job"]:
        fig_mae_grouped(t).savefig(FIGS / f"{t}.png", dpi=150)


if __name__ == "__main__":
    main()
