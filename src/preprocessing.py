"""Reading order, cleaning and masking of direct answers.

python src/preprocessing.py  -> data/processed/clean.csv with columns
id, job, sex, race, exp_year, text_raw, text_clean, text_lowconf, text_ref, masked_ocr
"""
import re

import numpy as np
import pandas as pd

from utils import CAP, OCR_DIR, PROCESSED, RAW, TABLES, load_json, save_json

CONF_THRESHOLD = 0.60  # lines below this are dropped in the "lowconf" variant

# --------------------------------------------------------------------------- reading order


def reading_order(lines, width=None):
    """Sort OCR lines top-to-bottom, left-to-right, handling two-column layouts.

    1. A vertical "gutter" is searched in the middle half of the page: the x with the fewest
       lines crossing it. If almost nothing crosses it and both sides hold a good share of the
       lines, the page is two-column.
    2. Lines crossing the gutter (headers, full-width lines) split the page into horizontal
       segments; inside each segment the left column is read completely, then the right one.
       This keeps a job title next to its dates, which live in the same column.
    3. Within a column, lines whose vertical centres are closer than 0.6 x the median line
       height are the same row and are sorted left to right.
    """
    if not lines:
        return []
    boxes = np.array([l["box"] for l in lines], dtype=float)
    W = width or boxes[:, 2].max()
    heights = boxes[:, 3] - boxes[:, 1]
    tol = 0.6 * float(np.median(heights))

    def sort_rows(idx):
        idx = sorted(idx, key=lambda i: (boxes[i, 1] + boxes[i, 3]) / 2)
        rows, cur, cur_y = [], [], None
        for i in idx:
            yc = (boxes[i, 1] + boxes[i, 3]) / 2
            if cur and abs(yc - cur_y) > tol:
                rows.append(cur)
                cur = []
            if not cur:
                cur_y = yc
            cur.append(i)
        if cur:
            rows.append(cur)
        return [i for r in rows for i in sorted(r, key=lambda k: boxes[k, 0])]

    xs = np.linspace(0.25 * W, 0.75 * W, 60)
    crossing = [int(((boxes[:, 0] < x) & (boxes[:, 2] > x)).sum()) for x in xs]
    gutter = xs[int(np.argmin(crossing))]
    n = len(lines)
    centers = (boxes[:, 0] + boxes[:, 2]) / 2
    left = (boxes[:, 2] <= gutter).sum()
    right = (boxes[:, 0] >= gutter).sum()
    two_col = min(crossing) <= 0.15 * n and min(left, right) >= 0.2 * n and len(lines) > 12
    if not two_col:
        return [lines[i] for i in sort_rows(range(n))]

    spanning = [i for i in range(n) if boxes[i, 0] < gutter < boxes[i, 2]]
    ordered, remaining = [], set(range(n)) - set(spanning)
    cuts = sorted(spanning, key=lambda i: boxes[i, 1])
    prev_y = -1
    for c in cuts + [None]:
        y_lim = boxes[c, 1] if c is not None else np.inf
        seg = [i for i in remaining if prev_y <= (boxes[i, 1] + boxes[i, 3]) / 2 < y_lim]
        ordered += sort_rows([i for i in seg if centers[i] < gutter])
        ordered += sort_rows([i for i in seg if centers[i] >= gutter])
        remaining -= set(seg)
        if c is not None:
            ordered.append(c)
            prev_y = boxes[c, 3]
    ordered += sort_rows(list(remaining))  # safety net
    return [lines[i] for i in ordered]


# --------------------------------------------------------------------------- cleaning

EMAIL = re.compile(r"[\w.+-]+\s*@\s*[\w-]+(\.[\w-]+)+")
PHONE = re.compile(r"(\+?\d{1,2}[\s.-]?)?(\(?\d{3}\)?[\s.-]?)\d{3}[\s.-]?\d{4}")
ADDRESS = re.compile(
    r"\b\d{1,5}\s+[A-Z][\w.]*(\s+[A-Z][\w.]*)*\s+(St|Street|Ave|Avenue|Rd|Road|Blvd|Dr|Drive|Ln|Lane|Ct|Way)\b\.?"
    r"|\b[A-Z][a-z]+(\s[A-Z][a-z]+)*,\s*[A-Z]{2}\s+\d{5}\b",
)
NAME_LINE = re.compile(r"^[A-Z][a-zA-Z'\-]+(\s+[A-Z]\.?)?(\s+[A-Z][a-zA-Z'\-]+){1,2}$")

# Direct answers: stated total experience, age, date of birth.
MASK_PATTERNS = [
    re.compile(r"\b\d{1,2}\s*\+?\s*(?:-|to)?\s*\d{0,2}\s*(?:years?|yrs?)\s*(?:of\s+)?(?:[a-z]+\s+){0,2}experience\b", re.I),
    re.compile(r"\bexperience\s*(?:of|:)?\s*\d{1,2}\s*\+?\s*(?:years?|yrs?)\b", re.I),
    re.compile(r"\b\d{1,2}\s*[- ]years?[- ]old\b", re.I),
    re.compile(r"\bage[ds]?\s*[:\-]?\s*\d{2}\b", re.I),
    re.compile(r"\b(?:date of birth|dob|born(?:\s+on|\s+in)?)\s*[:\-]?\s*[\w/ ,.\-]{4,20}\d{4}\b", re.I),
]
MASK_TOKEN = "[MASKED]"


def mask_direct_answers(text):
    n = 0
    for pat in MASK_PATTERNS:
        text, k = pat.subn(MASK_TOKEN, text)
        n += k
    return text, n


def clean_text(lines, min_conf=0.0):
    """Ordered OCR lines -> cleaned text (no PII, hyphenation fixed, answers masked).

    Returns (text, n_masked_phrases).
    """
    ordered = reading_order(lines)
    texts = [l["text"].strip() for l in ordered if l["score"] >= min_conf and l["text"].strip()]
    # the name sits in the first line(s) of a CV header
    texts = [t for k, t in enumerate(texts) if not (k < 3 and NAME_LINE.match(t))]
    merged = []
    for t in texts:
        if merged and merged[-1].endswith("-") and t[:1].islower():  # hyphenation
            merged[-1] = merged[-1][:-1] + t
        else:
            merged.append(t)
    text = "\n".join(merged)
    for pat in (EMAIL, PHONE, ADDRESS):
        text = pat.sub(" ", text)
    text, n = mask_direct_answers(text)
    return re.sub(r"[ \t]+", " ", text).strip(), n


def raw_text(lines):
    """Naive baseline: lines in the order PaddleOCR returned them, no cleaning."""
    return "\n".join(l["text"] for l in lines)


def clean_reference(text):
    """Reference text gets the same PII removal and masking so Q2 is comparable."""
    text = str(text)
    for pat in (EMAIL, PHONE, ADDRESS):
        text = pat.sub(" ", text)
    text, _ = mask_direct_answers(text)
    return re.sub(r"[ \t]+", " ", text).strip()


# --------------------------------------------------------------------------- main


def main():
    meta = pd.read_csv(RAW / "metadata.csv", dtype={"id": str})
    ref_f = RAW / "reference_text.csv"
    ref = pd.read_csv(ref_f, dtype={"id": str}).set_index("id")["text"].to_dict() if ref_f.exists() else {}
    if not ref:
        print("WARNING: no reference text; text_ref is empty (OCR quality / Q2 gap not meaningful)")
    rows = []
    for _, m in meta.iterrows():
        f = OCR_DIR / f"{m['id']}.json"
        if not f.exists():
            continue
        lines = load_json(f)["lines"]
        clean, n_masked = clean_text(lines)
        lowconf, _ = clean_text(lines, min_conf=CONF_THRESHOLD)
        rows.append({
            "id": m["id"], "job": m["job"], "sex": m["sex"], "race": m["race"],
            "exp_year_raw": m["exp_year"], "exp_year": float(min(m["exp_year"], CAP)),
            "text_raw": raw_text(lines), "text_clean": clean, "text_lowconf": lowconf,
            "text_ref": clean_reference(ref.get(m["id"], "")), "masked_ocr": n_masked,
        })
    df = pd.DataFrame(rows)
    PROCESSED.mkdir(parents=True, exist_ok=True)
    df.to_csv(PROCESSED / "clean.csv", index=False)

    stats = {
        "n_cvs": len(df),
        "cvs_with_masking": int((df["masked_ocr"] > 0).sum()),
        "phrases_masked": int(df["masked_ocr"].sum()),
        "share_at_cap": float((df["exp_year_raw"] >= CAP).mean()),
        "conf_threshold_lowconf_variant": CONF_THRESHOLD,
    }
    save_json(stats, TABLES.parent / "metrics" / "preprocessing_stats.json")
    print(stats)
    # 50 random texts to read by hand (required audit of the masking rule)
    df.sample(min(50, len(df)), random_state=42)[["id", "text_clean"]].to_csv(
        TABLES / "audit_sample_50.csv", index=False)


if __name__ == "__main__":
    main()
