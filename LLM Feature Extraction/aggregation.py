from typing import List, Optional


def aggregate_lactate_chunk_results(chunk_results: List[dict]) -> dict:
    presents = [r.get("present") for r in chunk_results]
    values = [r.get("lactate_value") for r in chunk_results if isinstance(r.get("lactate_value"), (int, float))]

    if True in presents:
        final_present = True
    elif False in presents:
        final_present = False
    else:
        final_present = None

    max_value: Optional[float] = max(values) if values else None

    # choose first evidence from a positive chunk if possible
    evidence = ""
    for r in chunk_results:
        if r.get("present") is True and r.get("evidence_quote"):
            evidence = r["evidence_quote"]
            break

    if not evidence:
        for r in chunk_results:
            if r.get("evidence_quote"):
                evidence = r["evidence_quote"]
                break

    return {
        "final_present": final_present,
        "final_lactate_value": max_value,
        "final_evidence_quote": evidence,
        "n_chunks": len(chunk_results)
    }