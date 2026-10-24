"""Run PaddleOCR once on every sampled image and cache one JSON per CV in data/ocr/.

Each JSON: {"id", "time_s", "lines": [{"text", "score", "box": [x0, y0, x1, y1]}], "size": [w, h]}
Recognition model: PP-OCRv5_server_rec (accuracy matters more than speed: a misread digit in a
date changes the target; the mobile model is ~3x faster but less accurate on small digits).
"""
import argparse
import platform
import time

from tqdm import tqdm

from utils import IMAGES, METRICS, OCR_DIR, load_json, save_json

_OCR = None


def get_ocr(rec_model="PP-OCRv5_server_rec", det_model="PP-OCRv5_server_det"):
    global _OCR
    if _OCR is None:
        from paddleocr import PaddleOCR

        _OCR = PaddleOCR(
            lang="en",
            text_detection_model_name=det_model,
            text_recognition_model_name=rec_model,
            use_doc_orientation_classify=False,
            use_doc_unwarping=False,
            use_textline_orientation=False,
            enable_mkldnn=False,  # oneDNN crashes on CPU with Paddle 3.x (ConvertPirAttribute2RuntimeAttribute)
        )
    return _OCR


def run_ocr(image_path, ocr=None):
    """Return a list of {"text", "score", "box"} dicts for one image."""
    ocr = ocr or get_ocr()
    res = ocr.predict(str(image_path))[0]
    lines = []
    for t, s, b in zip(res["rec_texts"], res["rec_scores"], res["rec_boxes"]):
        lines.append({"text": t, "score": float(s), "box": [int(v) for v in b]})
    return lines


def hardware():
    try:
        import paddle

        if paddle.device.is_compiled_with_cuda():
            return f"GPU ({paddle.device.cuda.get_device_name(0)})"
    except Exception:
        pass
    return f"CPU ({platform.processor() or platform.machine()})"


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--limit", type=int, default=None)
    ap.add_argument("--rec-model", default="PP-OCRv5_server_rec")
    ap.add_argument("--det-model", default="PP-OCRv5_server_det")
    args = ap.parse_args()

    OCR_DIR.mkdir(parents=True, exist_ok=True)
    paths = sorted(IMAGES.glob("*.png"))[: args.limit]
    ocr = get_ocr(args.rec_model, args.det_model)
    failed = []
    for p in tqdm(paths):
        out = OCR_DIR / f"{p.stem}.json"
        if out.exists():  # cached: OCR is the slowest stage, never repeat it
            continue
        try:
            from PIL import Image

            size = list(Image.open(p).size)
            t0 = time.time()
            lines = run_ocr(p, ocr)
            save_json({"id": p.stem, "time_s": time.time() - t0, "size": size, "lines": lines}, out)
        except Exception as e:  # noqa: BLE001
            failed.append({"id": p.stem, "error": str(e)})

    docs = [load_json(f) for f in OCR_DIR.glob("*.json")]
    confs = [l["score"] for d in docs for l in d["lines"]]
    summary = {
        "n_cvs": len(docs),
        "failed": failed,
        "mean_time_per_cv_s": sum(d["time_s"] for d in docs) / max(len(docs), 1),
        "mean_confidence": sum(confs) / max(len(confs), 1),
        "rec_model": args.rec_model,
        "det_model": args.det_model,
        "hardware": hardware(),
    }
    save_json(summary, METRICS / "ocr_summary.json")
    print(summary)


if __name__ == "__main__":
    main()
