"""Streamlit app: python -m streamlit run app.py   (exactly 4 tabs)."""
import sys
from pathlib import Path

import pandas as pd
import streamlit as st

sys.path.insert(0, str(Path(__file__).parent / "src"))
from utils import CAP, IMAGES, METRICS, OCR_DIR, TABLES, load_json  # noqa: E402

st.set_page_config(page_title="CV experience estimator", layout="wide")
st.title("Predicting years of experience from CV images")


def table(name, **kw):
    f = TABLES / name
    if f.exists():
        return pd.read_csv(f, **kw)
    st.info(f"`{f.name}` not found yet. Run the pipeline (see README).")
    return None


def show(df, **kw):
    if df is not None:
        st.dataframe(df, use_container_width=True, **kw)


@st.cache_resource
def predictors():
    import baseline_embeddings, baseline_median, baseline_tfidf
    from train_bert import load_predictor

    out = {"Median": baseline_median.predict, "TF-IDF + Ridge": baseline_tfidf.predict,
           "Embeddings + Ridge": baseline_embeddings.predict}
    try:
        out["Fine-tuned BERT"] = load_predictor()
    except Exception as e:  # noqa: BLE001
        st.warning(f"BERT model not available: {e}")
    return out


tab1, tab2, tab3, tab4 = st.tabs(["Problem & Methodology", "OCR & Exploration", "Model Results", "Estimator"])

with tab1:
    st.header("Objective")
    st.write(
        "Estimate a candidate's years of work experience (0-20) from the **image** of a CV. The number is rarely "
        "written: it has to be inferred from job dates, the number of positions, title progression and graduation "
        "year. A recruiter does this mentally; the model has to learn it from text.")
    st.header("Data and target")
    st.write(
        f"PopResume (synthetic, CC BY 4.0). 800 no-photo CVs for each of 5 occupations (4,000), fixed seed 42. "
        f"Target `exp_year` clipped to 0-{CAP} (\"{CAP} or more\"). The reference text is used only to measure OCR "
        "error and for the clean-text comparison, never as a training input. Age, sex, race, names and location are "
        "never features. Phrases such as \"12 years of experience\", age or birth dates are masked with regexes.")
    st.header("OCR")
    st.write(
        "PaddleOCR 3.x, PP-OCRv5 server detection + recognition (accuracy over speed: a misread digit changes the "
        "target). Reading order is rebuilt from boxes: rows by vertical centre, left to right; a vertical gutter is "
        "detected for two-column layouts and each column is read whole between full-width header lines, so a job "
        "title stays next to its dates.")
    st.header("Models and split")
    st.write(
        "70/15/15 split by CV id, stratified by occupation and experience band. Four models on the same split: "
        "median, TF-IDF + Ridge, frozen sentence embeddings + Ridge, and a fine-tuned DistilBERT regressor "
        "(target / 20, MSE loss, head+tail truncation, early stopping on validation MAE).")
    st.header("Limitations")
    st.write("Synthetic resumes, 5 occupations, U.S. statistics, English only. Not suitable for deciding who is hired.")

with tab2:
    st.header("Distribution of experience")
    f = Path(__file__).parent / "output" / "figures" / "distribution.png"
    if f.exists():
        st.image(str(f))
    st.caption("Histogram shows the shape and how many CVs sit at the cap; boxplots compare occupations.")
    st.header("OCR quality")
    show(table("ocr_quality_by_job.csv"))
    f = Path(__file__).parent / "output" / "figures" / "ocr_quality.png"
    if f.exists():
        st.image(str(f))
    st.caption("CER/WER histograms show the error distribution; 'years read correctly' is the share of reference "
               "years found in the OCR text, the figure that matters most for this task.")
    st.header("Image vs extracted text")
    docs = sorted(OCR_DIR.glob("*.json"))
    if docs:
        pick = st.selectbox("CV", [d.stem for d in docs])
        img_path = IMAGES / f"{pick}.png"
        c1, c2 = st.columns(2)
        doc = load_json(OCR_DIR / f"{pick}.json")
        if img_path.exists():
            from PIL import Image, ImageDraw

            img = Image.open(img_path).convert("RGB")
            dr = ImageDraw.Draw(img)
            for l in doc["lines"]:
                dr.rectangle(l["box"], outline="red", width=2)
            c1.image(img, caption="PaddleOCR boxes")
        from preprocessing import clean_text

        c2.text_area("Cleaned text (reading order rebuilt)", clean_text(doc["lines"])[0], height=500)

with tab3:
    st.header("Model comparison (test set)")
    show(table("model_comparison.csv"))
    st.header("BERT's MAE gain (95% paired bootstrap CI)")
    show(table("bert_gain.csv"))
    names = {"Median": "median", "TF-IDF + Ridge": "tfidf", "Embeddings + Ridge": "embeddings", "Fine-tuned BERT": "bert"}
    sel = st.selectbox("Model", list(names))
    fig_dir = Path(__file__).parent / "output" / "figures"
    c1, c2 = st.columns(2)
    for col, pre in [(c1, "pred_vs_true"), (c2, "confusion")]:
        f = fig_dir / f"{pre}_{names[sel]}.png"
        if f.exists():
            col.image(str(f))
    st.caption("Points far from the dashed 45° line are errors; a flat cloud means predictions are pulled to the middle.")
    st.header("MAE by band and occupation")
    c1, c2 = st.columns(2)
    with c1:
        show(table("mae_by_band.csv"))
    with c2:
        show(table("mae_by_job.csv"))
    st.header("OCR noise cost (Q2): OCR text vs reference text")
    show(table("ocr_gap.csv"))
    st.header("Sensitivity (Q3)")
    show(table("sensitivity.csv"))
    st.header("Date-shift probe (Q1)")
    show(table("probe_shift_dates.csv"))
    st.header("Fairness check (Q4)")
    show(table("fairness_bert.csv"))
    st.header("Ten largest BERT errors")
    show(table("top10_bert_errors.csv"))

with tab4:
    st.header("Estimate experience from a CV")
    up = st.file_uploader("CV image or PDF", type=["png", "jpg", "jpeg", "pdf"])
    if up is not None:
        from PIL import Image

        if up.name.lower().endswith(".pdf"):
            import pypdfium2 as pdfium

            img = pdfium.PdfDocument(up.read())[0].render(scale=2).to_pil().convert("RGB")
        else:
            img = Image.open(up).convert("RGB")
        st.image(img, width=400)
        tmp = Path("_upload.png")
        img.save(tmp)
        with st.spinner("Running PaddleOCR..."):
            from ocr_extraction import run_ocr
            from preprocessing import clean_text

            text, _ = clean_text(run_ocr(tmp))
        st.text_area("Extracted text", text, height=300)
        preds = {n: float(f([text])[0]) for n, f in predictors().items()}
        st.dataframe(pd.DataFrame({"Model": preds.keys(), "Predicted years": [round(v, 1) for v in preds.values()]}))
    st.header("Browse test CVs")
    pr = METRICS / "preds_bert.csv"
    if pr.exists():
        from evaluation import load_test

        te = load_test()
        te["abs_err"] = (te["bert"] - te["exp_year"]).abs()
        jobs = st.multiselect("Occupation", sorted(te["job"].unique()), default=sorted(te["job"].unique()))
        yr = st.slider("True years", 0, CAP, (0, CAP))
        er = st.slider("Minimum BERT error (years)", 0.0, 15.0, 0.0)
        view = te[te["job"].isin(jobs) & te["exp_year"].between(*yr) & (te["abs_err"] >= er)]
        st.dataframe(view[["id", "job", "exp_year", "median", "tfidf", "embeddings", "bert", "abs_err"]].round(2))
        st.subheader("Best and worst BERT predictions")
        c1, c2 = st.columns(2)
        c1.write("Best (smallest error)")
        c1.dataframe(te.nsmallest(5, "abs_err")[["id", "exp_year", "bert", "abs_err"]].round(2))
        c2.write("Worst (largest error)")
        c2.dataframe(te.nlargest(5, "abs_err")[["id", "exp_year", "bert", "abs_err"]].round(2))
        st.caption("Large errors typically come from misread dates, overlapping jobs, career gaps or truncated text "
                   "(see the README error analysis).")
    else:
        st.info("Run the pipeline to browse test CVs.")
