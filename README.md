# LLM-Based Clinical Feature Extraction from Discharge Notes

**Validation and Retrospective Risk Analysis in a Post-Cardiac Arrest Cohort**

Master thesis — Çağan Yiğit Deliktaş
Data and Web Science Group, University of Mannheim (Prof. Dr. Philipp Kellmeyer)

This repository holds the code for my master thesis. The project asks whether a
large language model can turn free-text discharge summaries into a small set of
explicit, clinically interpretable features for patients admitted to intensive
care after cardiac arrest, and what those features reveal when compared against
an early structured estimate of mortality risk.

Instead of representing clinical text as opaque embedding vectors, the pipeline
extracts three named clinical concepts, each one together with the sentence from
the note that supports it, so a clinician could in principle audit every label.

## Research questions

- **RQ1** — Can an LLM reliably extract clinically interpretable features from
  discharge notes, as validated against manual annotation?
- **RQ2** — Do the extracted features line up with early structured mortality
  risk in clinically expected ways (do patients with documented severity
  indicators sit at higher baseline risk)?
- **RQ3** — What does the *indeterminate* ("unknown") output of the extraction
  represent: genuinely no evidence in the note, incomplete documentation of a
  state that did happen, or a clinically distinct subgroup with its own risk
  profile?

## The three extracted features

Three features were chosen because they span three different extraction regimes:

| Feature | Regime | Output |
|---|---|---|
| **Lactate** | numeric | elevated / not / indeterminate, plus the max numeric value |
| **Shock** | latent, ordinal | present / not / indeterminate, plus severity (none/mild/moderate/severe) |
| **Coma / unresponsiveness** | binary | present / not / indeterminate |

Every feature produces a three-class presence label: **evidence detected**
(positive), **no evidence detected** (negative), and **indeterminate** (null).
The indeterminate class is kept as its own category on purpose, rather than
folded into the negative one, because a central finding of the thesis is that
this uncertainty carries feature-specific clinical meaning (for coma it marks a
small but severely ill, high-mortality subgroup).

## Data

No patient data is included in this repository. Both MIMIC-IV
([Johnson et al., 2023]) and MIMIC-IV-Note are available only through
credentialed access on PhysioNet, and the note license does not allow the
de-identified text to be sent to third-party services, so the raw tables, the
derived datasets and all model outputs are kept out of git (see `.gitignore`).
The code expects the data folders (`data/`, `Outputs/`, `outputs/`) to sit next
to the notebooks locally; they are not part of the repo.

Only discharge summaries are used (not radiology reports). The cohort is the
2,307 cardiac-arrest ICU patients reconstructed from MIMIC-IV; 1,618 of them had
a linkable discharge note and enter the note-based analyses.

## Repository structure

The repo follows the same order as the thesis: reconstruct the structured
baseline, extract the note features, validate them, then relate them to risk.

```
Preprocessing/
  Preprocessing.ipynb                     Structured pipeline: cohort + first-24h features
  Preprocessing_part_2_textual_data.ipynb Embedding baseline (TF-IDF / BERT) reconstructed from prior work
  inspect_notes.ipynb                     Exploring the discharge-note structure before writing the cleaner
  hadm_id_uniqueness.ipynb                Sanity check on the admission id

LLM Feature Extraction/          -- the extraction pipeline (thesis Ch. 5)
  main.py                    Entry point: run one feature over the whole cohort
  config.py                  Paths, chunking settings, keyword lists, per-feature configuration
  note_processing.py         Text cleaning, sentence splitting, keyword selection, chunking
  prompts.py                 Feature-specific prompts (lactate, shock, coma)
  llm_client.py              Client for the local llama.cpp server
  parsing.py                 Parsing and normalising the model's JSON replies
  aggregation.py             Rolling up the per-chunk answers into one label per patient
  result_builders.py         Building the final per-note output rows
  check_unique_hadm_id_in_cohort.ipynb    Cohort sanity check
  Result Analysis/           Validation: manual annotation and the "gold rule" checks
    manual_annotation_sample_selection.ipynb   Pick + display the notes to annotate
    manual_annotation_result_evaluation.ipynb  LLM vs. manual labels per feature
    figures_manual_annotation.ipynb            Confusion matrices / agreement figures
    first_analysis_gold_rule.ipynb             Structured-vs-LLM lactate, shock added value
    coma_second_prompt.ipynb                   Second-iteration coma prompt comparison

Modelling/                       -- baseline risk and retrospective analysis (thesis Ch. 4, 6)
  structured_model_first24h.ipynb            Leakage-aware CV, out-of-fold baseline risk, unknown-category tests
  Modelling_part_3_structured_dataset.ipynb  Feature selection (LASSO, XGBoost) + logistic regression, structured
  Modelling_part_3_textual_dataset.ipynb     Same pipeline with the textual features added
  coma_unknown_exploratory_analysis_FINAL.ipynb  Why the coma-indeterminate group is high-risk (GCS, RASS, sedation, ventilation)
  figures_risk_and_profiles.ipynb            Thesis figures: risk separation, mortality, severity, physiology, GCS
```

## Structured baseline risk

The structured side reconstructs the cardiac-arrest pipeline of prior work
(cohort, 18 first-24h predictors selected by LASSO and XGBoost) and adds one
methodological change: all imputation and scaling are moved **inside** a
stratified five-fold cross-validation loop, so nothing about a held-out patient
leaks into their own preprocessing. A logistic regression is fit per fold, and
each patient is scored by the fold that did not train on them. The resulting
out-of-fold probability (`baseline_risk_oof`) is used as a per-patient baseline
mortality risk (out-of-fold AUROC ≈ 0.78, Brier ≈ 0.19, well calibrated) that
the note-derived features are compared against.

## LLM extraction pipeline

The extraction runs against a local model served through
[llama.cpp](https://github.com/ggerganov/llama.cpp) via its OpenAI-compatible
chat endpoint, so the note text never leaves the machine. Each note goes through
five stages:

1. **Text cleaning** — remove the de-identification placeholders (`[**...**]`,
   `___`) and fix wrapped lines, while keeping lab-result lines intact
   (`note_processing.py`).
2. **Keyword sentence selection** — keep only the sentences near a feature
   keyword, plus a one-sentence context window, which cuts down how much text
   the model sees.
3. **Chunking** — pack the selected sentences into overlapping chunks of five
   sentences (overlap of one).
4. **Chunk-level extraction** — send each chunk with a feature-specific prompt
   that asks for a strict JSON answer (presence label, evidence quote(s), and
   the numeric/ordinal field where relevant); replies are parsed and normalised
   (`prompts.py`, `llm_client.py`, `parsing.py`).
5. **Patient-level aggregation** — combine the per-chunk answers into one record
   per patient. The rules are simple: any positive wins for presence, the max
   for lactate value, the worst (consistent) level for shock severity, and a
   supporting quote is carried through (`aggregation.py`).

The whole thing is driven by a feature-configuration dictionary in `config.py`,
so adding a feature means adding keywords, a prompt, a schema and an aggregator
rather than touching the pipeline.

### Model and setup

- **Model:** Meta Llama 3.1 8B-Instruct, 4-bit GGUF quantisation (`Q4_K_M`).
- **Server:** `llama.cpp` at `http://127.0.0.1:8080`, 4096-token context.
- **Decoding:** temperature `0.0` (deterministic), max 300 tokens per chunk,
  60-second request timeout.
- Runs fully offline on a laptop (tested on an Apple M3 Pro). Every per-chunk
  input and raw reply is logged, so any patient-level label can be traced back
  to the chunk, prompt and JSON that produced it.

### Running the extraction

```bash
# 1. start the llama.cpp server on the URL set in config.py
# 2. choose the feature at the bottom of main.py: main("lactate" | "shock" | "coma")
cd "LLM Feature Extraction"
python main.py
```

Two CSVs are written to `outputs/`: a patient-level file with the final labels
and a chunk-level file with the raw model output for every chunk (used for the
manual-annotation comparison and for debugging).

## Requirements

Written for Python 3.12. Main libraries:

- `pandas`, `numpy`, `scipy`
- `scikit-learn`, `xgboost`, `statsmodels`
- `matplotlib`, `seaborn`
- `requests` (LLM client) and a running `llama.cpp` server for the extraction
- `transformers` / BERT for the textual embedding baseline

## Author

Çağan Yiğit Deliktaş — University of Mannheim

[Johnson et al., 2023]: https://physionet.org/content/mimiciv/
