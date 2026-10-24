"""OCR quality against the reference text: CER, WER and share of years read correctly."""
import re
from collections import Counter

import jiwer
import pandas as pd

from utils import PROCESSED, TABLES, METRICS, load_json, save_json, OCR_DIR, RAW

YEAR = re.compile(r"\b(?:19|20)\d{2}\b")


def normalize(text):
    """Same normalisation for both sides: lowercase, drop markdown/punctuation noise, one space."""
    text = str(text).lower()
    text = re.sub(r"[#*_`|]", " ", text)
    text = re.sub(r"[^\w\s]", " ", text)
    return re.sub(r"\s+", " ", text).strip()


def years_correct(ref, hyp):
    """Share of reference years (multiset) that appear in the OCR text."""
    r, h = Counter(YEAR.findall(ref)), Counter(YEAR.findall(hyp))
    total = sum(r.values())
    return sum(min(c, h[y]) for y, c in r.items()) / total if total else float("nan")


def main():
    meta = pd.read_csv(RAW / "metadata.csv", dtype={"id": str})
    ref = pd.read_csv(RAW / "reference_text.csv", dtype={"id": str}).set_index("id")["text"]
    rows = []
    for _, m in meta.iterrows():
        f = OCR_DIR / f"{m['id']}.json"
        if not f.exists() or m["id"] not in ref.index:
            continue
        hyp = "\n".join(l["text"] for l in load_json(f)["lines"])  # raw order, unmasked
        r, h = normalize(ref[m["id"]]), normalize(hyp)
        rows.append({
            "id": m["id"], "job": m["job"],
            "cer": jiwer.cer(r, h), "wer": jiwer.wer(r, h),
            "years_correct": years_correct(str(ref[m["id"]]), hyp),
        })
    df = pd.DataFrame(rows)
    df.to_csv(TABLES / "ocr_quality_per_cv.csv", index=False)
    by_job = df.groupby("job")[["cer", "wer", "years_correct"]].mean()
    by_job.loc["ALL"] = df[["cer", "wer", "years_correct"]].mean()
    by_job.to_csv(TABLES / "ocr_quality_by_job.csv")
    save_json(by_job.loc["ALL"].to_dict(), METRICS / "ocr_quality.json")
    print(by_job)


if __name__ == "__main__":
    main()
