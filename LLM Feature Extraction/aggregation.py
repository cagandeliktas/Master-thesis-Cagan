"""
Combine the per-chunk model outputs into one answer per note.

Each note is split into several chunks and the model labels each chunk on its
own, so a note ends up with a list of chunk results that can disagree. The
functions here roll those up into a single decision. The general rule is
"any positive wins": if any chunk says the feature is present the note is
labelled present, and severity/lactate values take the worst/highest seen. The
matching evidence quote and justification are pulled from a positive chunk when
there is one.
"""

from typing import List, Dict, Any, Optional, Callable


def aggregate_present_rule(chunk_results: List[dict], field_name: str = "present") -> Optional[bool]:
    """Decide the note-level presence flag from all chunk flags.

    Returns True if any chunk was positive, False if any chunk was explicitly
    negative (and none positive), and None if no chunk gave a clear answer.
    """
    presents = [r.get(field_name) for r in chunk_results]

    if True in presents:
        return True
    elif False in presents:
        return False
    return None


def select_evidence(chunk_results: List[dict]) -> str:
    """Pick one evidence quote for the note (single-quote schema).

    Prefers a quote from a chunk that was labelled present; if there is none,
    falls back to the first non-empty quote. Returns an empty string if no
    chunk had a quote.
    """
    for r in chunk_results:
        if r.get("present") is True and r.get("evidence_quote"):
            return r["evidence_quote"]

    for r in chunk_results:
        if r.get("evidence_quote"):
            return r["evidence_quote"]

    return ""


def select_evidence_quotes(
    chunk_results: List[dict],
    present_field: str = "shock_present"
) -> List[str]:
    """Pick the evidence quotes for the note (multi-quote shock schema).

    Same idea as select_evidence but returns a list: it takes the quotes from
    the first positive chunk that has any, otherwise the first chunk with quotes
    at all, and cleans up whitespace/empty entries.
    """
    def normalize_quotes(value):
        if isinstance(value, list):
            return [str(q).strip() for q in value if str(q).strip()]
        if isinstance(value, str) and value.strip():
            return [value.strip()]
        return []

    for r in chunk_results:
        if r.get(present_field) is True:
            quotes = normalize_quotes(r.get("evidence_quotes"))
            if quotes:
                return quotes

    for r in chunk_results:
        quotes = normalize_quotes(r.get("evidence_quotes"))
        if quotes:
            return quotes

    return []


def aggregate_feature_chunk_results(
        chunk_results: List[dict],
        extra_aggregations: Optional[Dict[str, Callable[[List[dict]], Any]]] = None
) -> Dict[str, Any]:
    """Generic roll-up used by the simple features (lactate, coma).

    Produces the note-level presence flag, one evidence quote and the chunk
    count. Any feature-specific extra fields (like the max lactate value) can be
    passed in through extra_aggregations as {output_key: function}.
    """
    result = {
        "final_present": aggregate_present_rule(chunk_results),
        "final_evidence_quote": select_evidence(chunk_results),
        "n_chunks": len(chunk_results)
    }

    if extra_aggregations:
        for output_key, func in extra_aggregations.items():
            result[output_key] = func(chunk_results)

    return result


def aggregate_max_numeric(chunk_results: List[dict], field_name: str) -> Optional[float]:
    """Return the highest numeric value of a field across chunks, or None."""
    values = [
        r.get(field_name)
        for r in chunk_results
        if isinstance(r.get(field_name), (int, float))
    ]
    return max(values) if values else None


def aggregate_lactate_chunk_results(chunk_results: List[dict]) -> dict:
    """Roll up lactate chunks: presence, evidence, and the max lactate value."""
    return aggregate_feature_chunk_results(
        chunk_results,
        extra_aggregations={
            "final_lactate_value": lambda results: aggregate_max_numeric(results, "lactate_value")
        }
    )


SEVERITY_ORDER = {
    "none": 0,
    "mild": 1,
    "moderate": 2,
    "severe": 3
}


def aggregate_worst_severity(
        chunk_results: List[dict],
        field_name: str = "severity"
) -> Optional[str]:
    """Return the most severe severity label seen across chunks, or None.

    Uses SEVERITY_ORDER to rank none < mild < moderate < severe.
    """
    severities = [
        r.get(field_name)
        for r in chunk_results
        if r.get(field_name) in SEVERITY_ORDER
    ]

    if not severities:
        return None

    return max(severities, key=lambda x: SEVERITY_ORDER[x])


def aggregate_shock_chunk_results_old(chunk_results: List[dict]) -> dict:
    """Earlier shock roll-up, kept for reference.

    Works with the first shock prompt schema (present/severity/evidence_quote).
    The pipeline now uses aggregate_shock_chunk_results with the newer schema.
    """
    return aggregate_feature_chunk_results(
        chunk_results,
        extra_aggregations={
            "final_severity": lambda results: aggregate_worst_severity(results, "severity")
        }
    )


def aggregate_shock_chunk_results(chunk_results: List[dict]) -> dict:
    """Roll up shock chunks into the final per-note shock result.

    Decides presence with the any-positive rule, then picks the worst severity
    that is consistent with that presence flag (mild/moderate/severe when
    present, none when absent). It also selects the supporting evidence quotes,
    the justification from a positive chunk when available, and the highest
    confidence score seen.
    """
    final_present = aggregate_present_rule(chunk_results, field_name="shock_present")

    if final_present is True:
        valid_severities = {"mild", "moderate", "severe"}
    elif final_present is False:
        valid_severities = {"none"}
    else:
        valid_severities = set()

    severities = [
        r.get("shock_severity")
        for r in chunk_results
        if r.get("shock_severity") in valid_severities
    ]

    final_severity = (
        max(severities, key=lambda x: SEVERITY_ORDER[x]) if severities else None
    )

    final_evidence_quotes = select_evidence_quotes(chunk_results, present_field="shock_present")

    final_justification = ""
    for r in chunk_results:
        if r.get("shock_present") is True and r.get("justification"):
            final_justification = r["justification"]
            break
    if not final_justification:
        for r in chunk_results:
            if r.get("justification"):
                final_justification = r["justification"]
                break

    confidences = [
        r.get("confidence")
        for r in chunk_results
        if isinstance(r.get("confidence"), (int, float))
    ]
    final_confidence = max(confidences) if confidences else None

    return {
        "final_present": final_present,
        "final_severity": final_severity,
        "final_evidence_quotes": final_evidence_quotes,
        "final_justification": final_justification,
        "final_confidence": final_confidence,
        "n_chunks": len(chunk_results),
    }


def aggregate_coma_chunk_results(chunk_results: List[dict]) -> dict:
    """Roll up coma chunks: presence flag and one evidence quote."""
    return aggregate_feature_chunk_results(chunk_results)

