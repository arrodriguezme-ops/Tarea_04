"""Fine-tune a small BERT-type encoder as a regressor (single output, target / CAP).

Long-text strategies (CVs often exceed 512 tokens; the education block and the oldest jobs
are at the END of the CV, so plain head truncation loses the start dates):
  head       first max_len tokens
  head_tail  first 40% + last 60% of the token budget  (default)
  chunk      all windows of max_len tokens; trained per chunk, predictions averaged per CV
"""
import argparse
import time

import numpy as np
import pandas as pd
import torch
from torch.utils.data import DataLoader
from transformers import AutoModelForSequenceClassification, AutoTokenizer, get_linear_schedule_with_warmup

from utils import (CAP, CLEAN_TEXT_COL, METRICS, MODELS, REF_TEXT_COL, clip_years, save_json, set_seed,
                   split_with_text, SEED)

NAME = "distilbert-base-uncased"


def encode_ids(tok, text, max_len, strategy):
    """Return a list of token-id windows (without special tokens added yet)."""
    ids = tok(text, add_special_tokens=False)["input_ids"]
    room = max_len - 2
    if len(ids) <= room:
        return [ids]
    if strategy == "head":
        return [ids[:room]]
    if strategy == "head_tail":
        h = int(room * 0.4)
        return [ids[:h] + ids[-(room - h):]]
    step = room // 2  # chunk: 50% overlap
    return [ids[i:i + room] for i in range(0, len(ids) - room + step, step)]


def make_batch(tok, windows):
    feats = [tok.build_inputs_with_special_tokens(w) for w in windows]
    L = max(len(f) for f in feats)
    ids = torch.tensor([f + [tok.pad_token_id] * (L - len(f)) for f in feats])
    return ids, (ids != tok.pad_token_id).long()


@torch.no_grad()
def predict_texts(model, tok, texts, max_len, strategy, device, bs=32):
    """Mean over windows of each text, back to years, clipped."""
    model.eval()
    wins, owner = [], []
    for i, t in enumerate(texts):
        for w in encode_ids(tok, str(t), max_len, strategy):
            wins.append(w)
            owner.append(i)
    preds = []
    for s in range(0, len(wins), bs):
        ids, mask = make_batch(tok, wins[s:s + bs])
        out = model(input_ids=ids.to(device), attention_mask=mask.to(device)).logits.squeeze(-1)
        preds += out.float().cpu().tolist()
    owner = np.array(owner)
    preds = np.array(preds)
    return clip_years(np.array([preds[owner == i].mean() for i in range(len(texts))]) * CAP)


def load_predictor(tag="main"):
    """Used by the app: returns f(list[str]) -> np.ndarray of years."""
    d = MODELS / "bert" / tag
    cfg = np.load(d / "cfg.npy", allow_pickle=True).item()
    dev = "cuda" if torch.cuda.is_available() else "cpu"
    tok = AutoTokenizer.from_pretrained(d)
    model = AutoModelForSequenceClassification.from_pretrained(d).to(dev)
    return lambda texts: predict_texts(model, tok, texts, cfg["max_len"], cfg["strategy"], dev)


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--model", default=NAME)
    ap.add_argument("--text-col", default=CLEAN_TEXT_COL)
    ap.add_argument("--strategy", default="head_tail", choices=["head", "head_tail", "chunk"])
    ap.add_argument("--loss", default="mse", choices=["mse", "huber"])
    ap.add_argument("--max-len", type=int, default=512)
    ap.add_argument("--lr", type=float, default=3e-5)
    ap.add_argument("--bs", type=int, default=16)
    ap.add_argument("--epochs", type=int, default=8)
    ap.add_argument("--patience", type=int, default=2)
    ap.add_argument("--tag", default="main")
    ap.add_argument("--seed", type=int, default=SEED)
    a = ap.parse_args()

    set_seed(a.seed)
    dev = "cuda" if torch.cuda.is_available() else "cpu"
    tok = AutoTokenizer.from_pretrained(a.model)
    model = AutoModelForSequenceClassification.from_pretrained(a.model, num_labels=1, problem_type="regression").to(dev)
    tr, va, te = (split_with_text(s) for s in ("train", "val", "test"))

    samples = []  # (token ids, scaled target)
    n_trunc = 0
    for t, y in zip(tr[a.text_col], tr["exp_year"]):
        ws = encode_ids(tok, str(t), a.max_len, a.strategy)
        n_trunc += len(tok(str(t), add_special_tokens=False)["input_ids"]) > a.max_len - 2
        samples += [(w, y / CAP) for w in ws]
    loader = DataLoader(samples, batch_size=a.bs, shuffle=True, collate_fn=lambda b: b)
    opt = torch.optim.AdamW(model.parameters(), lr=a.lr, weight_decay=0.01)
    total = len(loader) * a.epochs
    sched = get_linear_schedule_with_warmup(opt, int(0.1 * total), total)
    loss_fn = torch.nn.MSELoss() if a.loss == "mse" else torch.nn.HuberLoss(delta=2 / CAP)

    out_dir = MODELS / "bert" / a.tag
    best, bad, history = 1e9, 0, []
    t0 = time.time()
    for ep in range(a.epochs):
        model.train()
        for batch in loader:
            ids, mask = make_batch(tok, [w for w, _ in batch])
            y = torch.tensor([v for _, v in batch], dtype=torch.float)
            pred = model(input_ids=ids.to(dev), attention_mask=mask.to(dev)).logits.squeeze(-1)
            loss = loss_fn(pred, y.to(dev))
            loss.backward()
            torch.nn.utils.clip_grad_norm_(model.parameters(), 1.0)
            opt.step(); sched.step(); opt.zero_grad()
        val_mae = float(np.abs(predict_texts(model, tok, va[a.text_col], a.max_len, a.strategy, dev) - va["exp_year"]).mean())
        history.append({"epoch": ep + 1, "val_mae": val_mae})
        print(f"epoch {ep + 1}: val MAE {val_mae:.3f}")
        if val_mae < best:
            best, bad = val_mae, 0
            model.save_pretrained(out_dir); tok.save_pretrained(out_dir)
            np.save(out_dir / "cfg.npy", {"max_len": a.max_len, "strategy": a.strategy})
        else:
            bad += 1
            if bad >= a.patience:
                break
    train_s = time.time() - t0

    model = AutoModelForSequenceClassification.from_pretrained(out_dir).to(dev)  # best epoch
    t0 = time.time()
    p_ocr = predict_texts(model, tok, te[a.text_col], a.max_len, a.strategy, dev)
    inf = (time.time() - t0) / len(te)
    p_ref = predict_texts(model, tok, te[REF_TEXT_COL], a.max_len, a.strategy, dev)
    suffix = "" if a.tag == "main" else f"__{a.tag}"
    pd.DataFrame({"id": te["id"], "pred_ocr": p_ocr, "pred_ref": p_ref}).to_csv(
        METRICS / f"preds_bert{suffix}.csv", index=False)
    save_json({
        "train_s": train_s, "infer_s_per_cv": inf, "best_val_mae": best, "history": history,
        "hardware": torch.cuda.get_device_name(0) if dev == "cuda" else "CPU",
        "share_train_cvs_truncated": n_trunc / len(tr), **vars(a),
    }, METRICS / f"time_bert{suffix}.json")


if __name__ == "__main__":
    main()
