from typing import List, Dict, Any, Optional, Callable


def aggregate_present_rule(chunk_results: List[dict]) -> Optional[bool]:
    presents = [r.get("present") for r in chunk_results]

    if True in presents:
        return True
    elif False in presents:
        return False
    return None


def select_evidence(chunk_results: List[dict]) -> str:
    for r in chunk_results:
        if r.get("present") is True and r.get("evidence_quote"):
            return r["evidence_quote"]

    for r in chunk_results:
        if r.get("evidence_quote"):
            return r["evidence_quote"]

    return ""


def aggregate_feature_chunk_results(
    chunk_results: List[dict],
    extra_aggregations: Optional[Dict[str, Callable[[List[dict]], Any]]] = None
) -> Dict[str, Any]:
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
    values = [
        r.get(field_name)
        for r in chunk_results
        if isinstance(r.get(field_name), (int, float))
    ]
    return max(values) if values else None


def aggregate_lactate_chunk_results(chunk_results: List[dict]) -> dict:
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
    severities = [
        r.get(field_name)
        for r in chunk_results
        if r.get(field_name) in SEVERITY_ORDER
    ]

    if not severities:
        return None

    return max(severities, key=lambda x: SEVERITY_ORDER[x])


def aggregate_shock_chunk_results(chunk_results: List[dict]) -> dict:
    return aggregate_feature_chunk_results(
        chunk_results,
        extra_aggregations={
            "final_severity": lambda results: aggregate_worst_severity(results, "severity")
        }
    )