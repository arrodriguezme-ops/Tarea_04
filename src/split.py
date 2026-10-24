"""70/15/15 split by CV id, stratified by occupation x experience band, fixed seed."""
import pandas as pd
from sklearn.model_selection import train_test_split

from utils import SEED, SPLITS, load_processed, to_band


def main():
    df = load_processed()[["id", "job", "exp_year"]].copy()
    df["band"] = to_band(df["exp_year"])
    strat = df["job"] + "|" + df["band"]
    # merge strata too small to split (< 3 members) into a per-job bucket
    small = strat.map(strat.value_counts()) < 3
    strat[small] = df.loc[small, "job"]
    train, rest = train_test_split(df, test_size=0.30, stratify=strat, random_state=SEED)
    val, test = train_test_split(rest, test_size=0.50, stratify=strat[rest.index], random_state=SEED)
    SPLITS.mkdir(parents=True, exist_ok=True)
    for name, part in [("train", train), ("val", val), ("test", test)]:
        part.sort_values("id").to_csv(SPLITS / f"{name}.csv", index=False)
        print(name, len(part))


if __name__ == "__main__":
    main()
