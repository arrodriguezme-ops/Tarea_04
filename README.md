# cv_experience_ocr_bert

Predict a candidate's **years of work experience (0-20)** from the *image* of a CV:
PaddleOCR reads the CV, then four models guess the number (median, TF-IDF + Ridge, frozen sentence embeddings + Ridge, fine-tuned DistilBERT regressor), all on the same split.
For students/analysts comparing what each model really uses, and as a prototype for recruiting teams that pre-fill profiles.

> **Status:** the code for every stage is written; the numbers below are **placeholders (`TBD`) until the pipeline is run** on the 4,000-CV sample. Fill them from `output/tables/` and `output/metrics/` after running the steps in "Reproduce". Nothing in this README should be read as a result until then.

## Pipeline

| Stage | Script | Output |
|---|---|---|
| Sampling | `src/download_data.py` | `data/raw/images/`, `metadata.csv`, `sampled_ids.txt`, `reference_text.csv` |
| OCR | `src/ocr_extraction.py` | `data/ocr/<id>.json` (text, confidence, box per line), `output/metrics/ocr_summary.json` |
| OCR quality | `src/ocr_quality.py` | `output/tables/ocr_quality_*.csv` |
| Cleaning / masking | `src/preprocessing.py` | `data/processed/clean.csv`, `output/metrics/preprocessing_stats.json` |
| Split | `src/split.py` | `data/splits/{train,val,test}.csv` |
| Baselines | `src/baseline_*.py` | `models/baselines/`, `output/metrics/preds_*.csv` |
| BERT | `src/train_bert.py` | `models/bert/`, `output/metrics/preds_bert.csv` |
| Evaluation | `src/evaluation.py`, `src/visualization.py` | `output/tables/`, `output/figures/` |

## Sampling
800 no-photo (`has_photo == False`) CVs per occupation (nurses, teachers, software developers, accountants, construction laborers) = 4,000, streamed from PopResume with seed 42; IDs saved in `data/raw/sampled_ids.txt`. IDs are `<occupation>_<id>` because raw IDs repeat across occupations. Target `exp_year` is clipped to 0-20 (the same cap for every model). Share at the cap: `TBD` (if > 15%, raise the cap in `src/utils.py:CAP` and justify). Only `sex` and `race` are kept, for the fairness check; age, names, state and region are dropped at download time and never used.

## OCR
PaddleOCR 3.x, `lang="en"`, **PP-OCRv5 server** detection + recognition (accuracy over speed: a misread digit changes the answer), orientation/unwarping/textline-orientation off (CVs are flat digital renders). Every line is cached with text, confidence and box. Reading order: rows are lines whose vertical centres differ by less than 0.6 x median line height, sorted left to right. Two-column layouts are detected by finding a vertical gutter that almost no line crosses; each column is then read whole between full-width header lines, so titles stay next to their dates. Time per CV / hardware / mean confidence / failures: `TBD` (see `output/metrics/ocr_summary.json`).

## OCR quality and cleaning
CER and WER via `jiwer` after identical normalisation (lowercase, punctuation/markdown removed), per CV and per occupation; "years read correctly" = share of reference 19xx/20xx years that appear in the OCR text. Results: CER `TBD`, WER `TBD`, years correct `TBD`.
Cleaning: reading-order rebuild, hyphenation repair, removal of name (header line), email, phone and address, and masking of stated total experience, age and date of birth (`[MASKED]`). Job dates and graduation years are kept on purpose. The mask patterns were checked on 50 random texts (`output/tables/audit_sample_50.csv`, **to be read by hand**); CVs affected: `TBD`. A "low-confidence removed" variant drops lines with score < 0.60 (a sensitivity choice, compared in Q3, not assumed better).

## Split
70/15/15 by CV id, stratified by occupation x band (0-2, 3-5, 6-10, 11-20), seed 42. All tuning on validation; test only for final evaluation.

## Models
- **Median**: training median.
- **TF-IDF + Ridge**: word 1-2-grams, numbers/years **kept** (they are the evidence), alpha tuned on validation MAE. The no-digits variant is the Q3 comparison.
- **Embeddings + Ridge**: frozen `all-MiniLM-L6-v2`; CVs are cut into overlapping 120-word chunks whose vectors are averaged; alpha tuned on validation.
- **BERT**: `distilbert-base-uncased`, `num_labels=1, problem_type="regression"`, target / 20, MSE (Huber optional), AdamW lr 3e-5, batch 16, max 8 epochs, early stopping on validation MAE (patience 2), seed 42, max length 512. Long CVs use **head + tail** (40% / 60% of the budget) because the oldest jobs and education are at the end of a CV; `head` and `chunk` are available for Q3. Training time / hardware / share of CVs truncated: `TBD` (`output/metrics/time_bert.json`).

## Results (test set) — `TBD`
`output/tables/model_comparison.csv` (MAE, RMSE, Acc ±1/2/3, band accuracy and macro-F1, R², train and inference time), `output/tables/bert_gain.csv` (MAE reduction vs each baseline with a 95% paired bootstrap CI, 2,000 resamples).

## Analytical questions — `TBD` (answer with your numbers)
1. **What does BERT learn?** Gains: `bert_gain.csv`; by band: `mae_by_band.csv`, `bias_by_band.csv` (pull to the middle); probe: `probe_shift_dates.csv` shifts every year of 20 test CVs by +5 (durations unchanged, so a date-reading model should barely move).
2. **Cost of OCR noise:** `ocr_quality_by_job.csv` and `ocr_gap.csv` (each model evaluated on OCR text and on reference text of the same test CVs; models are trained on OCR text).
3. **Sensitivity:** run the variants below; `sensitivity.csv` collects them.
4. **Business, fairness, ethics:** `fairness_bert.csv` (MAE and mean error by sex and race). Discuss uses (pre-filling a profile is acceptable; ranking or rejecting candidates is not), costs of ±2/±5 years and the asymmetry of over- vs underestimation, and limitations: synthetic resumes, 5 occupations, U.S. statistics, English only. The system must never decide who is hired.

The 10 largest BERT errors are in `top10_bert_errors.csv`; fill the `cause` column by inspecting each CV (OCR date misread, reading order, career gap, overlapping jobs, missing dates, truncation, other).

## Reproduce
```bash
pip install paddlepaddle==3.2.0 -i https://www.paddlepaddle.org.cn/packages/stable/cpu/   # or the GPU build
pip install -r requirements.txt
huggingface-cli login            # after accepting the PopResume access conditions
cd src
python download_data.py --per-job 800        # add --local-dir <folder> to reuse downloaded shards
python ocr_extraction.py                      # slow; cached, resumable (GPU recommended)
python ocr_quality.py
python preprocessing.py
python split.py
python baseline_median.py
python baseline_tfidf.py
python baseline_embeddings.py
python train_bert.py                          # GPU recommended
python evaluation.py
python visualization.py
# Q3 sensitivity examples
python baseline_tfidf.py --no-digits
python train_bert.py --strategy head --tag head
python train_bert.py --strategy chunk --tag chunk
python train_bert.py --loss huber --tag huber
python baseline_tfidf.py --text-col text_raw --tag __raw
python baseline_tfidf.py --text-col text_lowconf --tag __lowconf
python evaluation.py
```
Images, OCR JSON, processed CSVs and model weights are git-ignored; the commands above regenerate them. Put the video URL in `video/link.txt`.

## Streamlit app
```bash
python -m streamlit run app.py
```
Four tabs: Problem & Methodology, OCR & Exploration, Model Results, Interactive Estimator.

## Notes
- The repo folder should be named `cv_experience_ocr_bert` when submitted.
- `download_data.py` assumes the text config exposes `id`, `job` and `text` columns; adjust if the real schema differs.
- Work is organised in branches (`feature/data-sampling`, `feature/paddleocr-extraction`, `feature/baselines`, `feature/bert-regression`, `feature/evaluation`, `feature/streamlit-dashboard`) to be merged via Pull Requests.
