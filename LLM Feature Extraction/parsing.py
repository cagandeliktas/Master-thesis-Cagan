import json
from typing import Dict, List, Any


def _normalize_bool(value):
    if isinstance(value, bool) or value is None:
        return value

    if isinstance(value, str):
        v = value.strip().lower()
        if v == "true":
            return True
        if v == "false":
            return False
        if v == "null":
            return None

    return value


def parse_llm_json(raw_output: str, required_keys: List[str]) -> Dict[str, Any]:
    if not isinstance(raw_output, str) or not raw_output.strip():
        raise ValueError("Empty LLM output")

    raw_output = raw_output.strip()
    start = raw_output.find("{")

    if start == -1:
        raise ValueError(f"No JSON object found in output: {raw_output[:300]}")

    decoder = json.JSONDecoder()

    try:
        data, _ = decoder.raw_decode(raw_output[start:])
    except json.JSONDecodeError as e:
        raise ValueError(
            f"Could not parse first JSON object. Error: {e}. "
            f"Raw output starts with: {raw_output[start:start + 500]}"
        )

    for key in required_keys:
        if key not in data:
            data[key] = None

    # Normalize boolean-like fields
    for key in ["present", "shock_present"]:
        if key in data:
            data[key] = _normalize_bool(data[key])

    # Normalize evidence_quotes for shock schema
    if "evidence_quotes" in required_keys:
        eq = data.get("evidence_quotes")
        if eq is None:
            data["evidence_quotes"] = []
        elif isinstance(eq, str):
            data["evidence_quotes"] = [eq.strip()] if eq.strip() else []
        elif isinstance(eq, list):
            data["evidence_quotes"] = [str(x).strip() for x in eq if str(x).strip()]
        else:
            data["evidence_quotes"] = []

    # Normalize numeric confidence for shock schema
    if "confidence" in required_keys and "evidence_quotes" in required_keys:
        conf = data.get("confidence")
        try:
            data["confidence"] = float(conf) if conf is not None else None
        except (TypeError, ValueError):
            data["confidence"] = None

    return data