# Master Thesis — Cardiac Arrest Outcome Prediction with Structured and LLM-Extracted Features

This repository holds the code for my master thesis. The project studies
in-hospital mortality after cardiac arrest using the MIMIC-IV database, and asks
whether features pulled out of the free-text clinical notes by a large language
model add anything on top of the usual structured first-24h features.

The work has three parts that follow each other:

1. **Preprocessing** — build the cardiac arrest cohort from MIMIC-IV and turn the
   raw tables into a structured first-24h dataset, plus a set of textual features
   from the notes.
2. **LLM feature extraction** — run a locally hosted LLM over the discharge notes
   to extract three clinical features (elevated lactate, shock, and
   coma / unresponsiveness), then validate those against the structured data and
   against manual annotation.
3. **Modelling** — train the mortality models on the structured features, on the
   structured + textual features, and study how the LLM features (and their
   "unknown" cases) relate to risk. This part also produces the thesis figures.

## A note on the data

No patient data is included in this repository. MIMIC-IV is available only
through credentialed access on PhysioNet, so the raw tables, the derived
datasets and all model outputs are kept out of git (see `.gitignore`). The code
expects the data folders (`data/`, `Outputs/`, `outputs/`) to sit next to the
notebooks locally, but they are not part of the repo.

## Repository structure

```
Preprocessing/
  Preprocessing.ipynb                     Structured preprocessing (cohort + first-24h features)
  Preprocessing_part_2_textual_data.ipynb Textual features from notes (TF-IDF, BERT, SVD/PCA)
  inspect_notes.ipynb                     Exploring the discharge note structure
  hadm_id_uniqueness.ipynb                Sanity check on the admission id

LLM Feature Extraction/
  main.py                    Pipeline entry point (run one feature over the cohort)
  config.py                  Paths, chunking settings and per-feature configuration
  note_processing.py         Cleaning notes and splitting them into chunks
  prompts.py                 Prompt templates for lactate, shock and coma
  llm_client.py              Client for the local llama.cpp server
  parsing.py                 Parsing and cleaning the model's JSON replies
  aggregation.py             Rolling up per-chunk answers into one per note
  result_builders.py         Building the final per-note output rows
  check_unique_hadm_id_in_cohort.ipynb    Cohort sanity check
  Result Analysis/           Validation against structured data and manual annotation

Modelling/
  structured_model_first24h.ipynb           Structured mortality model + unknown-category analysis
  Modelling_part_3_structured_dataset.ipynb Feature selection and logistic regression (structured)
  Modelling_part_3_textual_dataset.ipynb    Same pipeline with textual features added
  coma_unknown_exploratory_analysis_FINAL.ipynb  Exploratory analysis of the coma-indeterminate group
  figures_risk_and_profiles.ipynb           Thesis figures (risk, mortality, severity, physiology, GCS)
```

## LLM feature extraction

The extraction runs against a local model served through
[llama.cpp](https://github.com/ggerganov/llama.cpp) with its OpenAI-compatible
chat endpoint, so the note text never leaves the machine. For each note the
pipeline:

1. cleans the raw MIMIC formatting and splits the note into sentences,
2. keeps the sentences near the feature keywords (plus some context) and packs
   them into short overlapping chunks,
3. sends each chunk to the model with a feature-specific prompt that asks for a
   strict JSON answer,
4. parses and normalizes the JSON, and
5. aggregates the per-chunk answers into a single label per note (the general
   rule is "any positive wins", and severity / lactate take the worst / highest
   value seen).

To run it, start the llama.cpp server (the URL is set in `config.py`), pick the
feature at the bottom of `main.py`, and run:

```bash
cd "LLM Feature Extraction"
python main.py   # set main("lactate"), main("shock") or main("coma") first
```

Two CSVs are written to `outputs/`: a patient-level file with the final labels
and a chunk-level file with the raw model output for each chunk.

## Requirements

The code was written in Python 3.12. The main libraries used across the
notebooks and scripts are:

- `pandas`, `numpy`, `scipy`
- `scikit-learn`, `xgboost`, `statsmodels`
- `matplotlib`, `seaborn`
- `requests` (for the LLM client) and a running `llama.cpp` server for the
  extraction step
- `transformers` / BERT for the textual embeddings

## Author

Çağan Deliktaş
