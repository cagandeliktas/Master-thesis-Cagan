import json
from typing import Dict, List, Any


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

    return data