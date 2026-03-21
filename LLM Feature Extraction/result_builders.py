import pandas as pd


def build_lactate_result_row(row: pd.Series, result: dict, chunks: list[str]) -> dict:
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
    return {
        "note_id": row["note_id"],
        "hadm_id": row["hadm_id"],
        "subject_id": row["subject_id"],
        "note_type": row["note_type"],
        "llm_present": result["final_present"],
        "llm_severity": result.get("final_severity"),
        "llm_evidence_quote": result["final_evidence_quote"],
        "n_chunks": result["n_chunks"],
        "chunks": chunks,
    }