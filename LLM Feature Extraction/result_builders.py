"""
Build the final per-note output rows.

After a note has been chunked, sent to the model and aggregated, these helpers
combine the patient identifiers from the original row with the aggregated LLM
result into a single flat dictionary. Each feature has its own builder because
the output columns differ (lactate also carries the structured lab value and a
derived structured label, shock carries severity and justification, etc.).
"""

import pandas as pd


def build_lactate_result_row(row: pd.Series, result: dict, chunks: list[str]) -> dict:
    """Build the output row for the lactate feature.

    Combines the note identifiers with the aggregated LLM result and also adds
    the structured lactate value from the row and a structured_label flag
    (True when the max structured lactate is above 2 mmol/L), which is used
    later to compare the LLM output against the structured data.
    """
    structured_label = None
    structured_lactate_max = row.get("structured_lactate_max")

    if pd.notna(structured_lactate_max):
        structured_label = bool(structured_lactate_max > 2)

    return {
        "note_id": row["note_id"],
        "hadm_id": row["hadm_id"],
        "subject_id": row["subject_id"],
        "note_type": row["note_type"],
        "llm_present": result["final_present"],
        "llm_lactate_value": result.get("final_lactate_value"),
        "llm_evidence_quote": result["final_evidence_quote"],
        "n_chunks": result["n_chunks"],
        "chunks": chunks,
        "structured_lactate_max": structured_lactate_max,
        "structured_label": structured_label,
    }


def build_shock_result_row(row: pd.Series, result: dict, chunks: list[str]) -> dict:
    """Build the output row for the shock feature.

    Carries the aggregated presence flag together with severity, the supporting
    evidence quotes, the model's justification and its confidence score.
    """
    return {
        "note_id": row["note_id"],
        "hadm_id": row["hadm_id"],
        "subject_id": row["subject_id"],
        "note_type": row["note_type"],
        "llm_present": result["final_present"],
        "llm_severity": result.get("final_severity"),
        "llm_evidence_quotes": result.get("final_evidence_quotes", []),
        "llm_justification": result.get("final_justification", ""),
        "llm_confidence": result.get("final_confidence"),
        "n_chunks": result["n_chunks"],
        "chunks": chunks,
    }


def build_coma_result_row(row: pd.Series, result: dict, chunks: list[str]) -> dict:
    """Build the output row for the coma / unresponsiveness feature.

    Carries the aggregated presence flag and the single best evidence quote.
    """
    return {
        "note_id": row["note_id"],
        "hadm_id": row["hadm_id"],
        "subject_id": row["subject_id"],
        "note_type": row["note_type"],
        "llm_present": result["final_present"],
        "llm_evidence_quote": result["final_evidence_quote"],
        "n_chunks": result["n_chunks"],
        "chunks": chunks,
    }