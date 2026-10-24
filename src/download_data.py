"""Stratified sampling of no-photo CV images from PopResume.

Usage:
    python src/download_data.py --per-job 800
    python src/download_data.py --per-job 800 --local-dir path/to/shards   # already downloaded shards

Writes data/raw/images/<job>_<id>.png, data/raw/metadata.csv (id, job, sex, race, exp_year)
and data/raw/sampled_ids.txt. Age, names, state and region are never saved.
sex and race are kept only for the fairness check.
"""
import argparse
import io
from pathlib import Path

import pandas as pd
from PIL import Image

from utils import IMAGES, JOBS, RAW, SEED, set_seed

KEEP = ["id", "job", "sex", "race", "exp_year"]


def resolve_prefixes(repo="sumin-yu/PopResume"):
    """Map our job keys to the real shard prefixes (data/images/<prefix>-NNNN.parquet)."""
    from huggingface_hub import HfFileSystem

    files = HfFileSystem().glob(f"datasets/{repo}/data/images/*.parquet")
    prefixes = sorted({Path(f).name.rsplit("-", 1)[0] for f in files})
    out = {}
    for job, kw in JOBS.items():
        match = [p for p in prefixes if kw in p]
        if len(match) != 1:
            raise SystemExit(f"Cannot resolve shard prefix for {job}: candidates {match} of {prefixes}")
        out[job] = match[0]
    return out


def sample_streaming(prefix, n, seed):
    from datasets import load_dataset

    ds = load_dataset(
        "parquet",
        data_files=f"hf://datasets/sumin-yu/PopResume/data/images/{prefix}-*.parquet",
        split="train",
        streaming=True,
    )
    ds = ds.filter(lambda r: not r["has_photo"]).shuffle(seed=seed, buffer_size=5000)
    for r in ds.take(n):
        img = r["image"]
        if isinstance(img, dict):
            img = Image.open(io.BytesIO(img["bytes"]))
        yield r, img


def sample_local(files, n, seed, allow_photo=False):
    import pyarrow.parquet as pq

    if not files:
        print("  no local shards for this job, skipping")
        return

    cols = ["id", "has_photo", "sex", "race", "exp_year", "image"]
    frames = [pq.read_table(f, columns=cols).to_pandas() for f in files]
    df = pd.concat(frames)
    if not allow_photo:
        df = df[~df["has_photo"]]
    if len(df) < n:
        print(f"  only {len(df)} no-photo rows locally (< {n})")
    for _, r in df.sample(min(n, len(df)), random_state=seed).iterrows():
        yield r, Image.open(io.BytesIO(r["image"]["bytes"]))


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--per-job", type=int, default=800)
    ap.add_argument("--seed", type=int, default=SEED)
    ap.add_argument("--local-dir", default=None, help="folder with <prefix>-NNNN.parquet shards")
    ap.add_argument("--jobs", nargs="*", default=None, help="restrict to these job keys")
    ap.add_argument("--allow-photo", action="store_true",
                    help="DEV ONLY (local mode): also sample photo CVs. Violates the assignment rules; "
                         "use just to test the pipeline, never for reported results.")
    ap.add_argument("--no-reference", action="store_true", help="skip downloading the text config")
    args = ap.parse_args()
    set_seed(args.seed)
    IMAGES.mkdir(parents=True, exist_ok=True)

    prefixes = None if args.local_dir else resolve_prefixes()
    rows = []
    for job, kw in JOBS.items():
        if args.jobs and job not in args.jobs:
            continue
        print(f"== {job}")
        if args.local_dir:
            import pyarrow.parquet as pq

            files = [f for f in sorted(Path(args.local_dir).glob(f"*{kw}*.parquet"))
                     if "has_photo" in pq.ParquetFile(f).schema_arrow.names]  # image shards only
            it = sample_local(files, args.per_job, args.seed, args.allow_photo)
        else:
            it = sample_streaming(prefixes[job], args.per_job, args.seed)
        for r, img in it:
            key = f"{job}_{r['id']}"
            img.convert("RGB").save(IMAGES / f"{key}.png")
            rows.append({"id": key, "job": job, "sex": r["sex"], "race": r["race"],
                         "exp_year": float(r["exp_year"])})
    meta = pd.DataFrame(rows, columns=KEEP)
    meta.to_csv(RAW / "metadata.csv", index=False)
    (RAW / "sampled_ids.txt").write_text("\n".join(meta["id"]), encoding="utf-8")
    print(meta.groupby("job").size())
    print(f"Share at/above cap: {(meta['exp_year'] >= 20).mean():.1%}")

    if args.no_reference:
        return
    # Reference text: used ONLY to measure OCR error and for the Question 2 "clean text" evaluation.
    if args.local_dir:  # <job>.parquet files of the "text" config (id, job, ..., text)
        import pyarrow.parquet as pq

        ref = pd.concat([pq.read_table(f, columns=["id", "job", "text"]).to_pandas()
                         for f in Path(args.local_dir).glob("*.parquet")
                         if "has_photo" not in pq.ParquetFile(f).schema_arrow.names])
    else:
        from datasets import load_dataset

        txt = load_dataset("sumin-yu/PopResume", "text")
        ref = pd.concat([pd.DataFrame(txt[s]) for s in txt])
    ref["id"] = ref["id"].astype(str)
    ref["key"] = ref["job"].astype(str) + "_" + ref["id"]
    ref = ref[ref["key"].isin(set(meta["id"]))][["key", "text"]].rename(columns={"key": "id"})
    ref.to_csv(RAW / "reference_text.csv", index=False)
    print(f"Reference text found for {len(ref)}/{len(meta)} CVs")


if __name__ == "__main__":
    main()
