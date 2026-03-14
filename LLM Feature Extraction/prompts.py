import textwrap


def build_lactate_prompt_old(note_chunk: str) -> str:
    return textwrap.dedent(f"""
    You are extracting one variable from a clinical note.

    Question:
    Is lactate elevated?

    Rules:
    - Use only the text in the note chunk.
    - present=true only if lactate is explicitly elevated/high, lactic acidosis is explicitly present, or a lactate value above 2.0 mmol/L is stated.
    - present=false only if lactate is explicitly normal/not elevated/cleared, or a lactate value of 2.0 mmol/L or below is stated.
    - present=null if lactate is not mentioned or cannot be determined.
    - Extract lactate_value if a numeric lactate value is stated.
    - units should be "mmol/L" when a lactate value is present and the units are mmol/L or implied as standard lactate lab units.
    - evidence_quote must be a short exact quote from the chunk.
    - Return JSON only.

    Return exactly one JSON object with this schema:
    {{
      "present": true or false or null,
      "lactate_value": number or null,
      "units": "mmol/L" or null,
      "evidence_quote": string,
      "confidence": "low" or "medium" or "high"
    }}

    NOTE CHUNK:
    {note_chunk}
    """).strip()


def build_lactate_prompt(note_chunk: str) -> str:
    user_prompt = textwrap.dedent(f"""
    Extract whether lactate is elevated from the note chunk.

    Rules:
    - present=true if lactate > 2.0 mmol/L or explicitly described as elevated.
    - present=false if lactate ≤ 2.0 mmol/L or explicitly described as normal/cleared.
    - present=null if lactate is not mentioned.

    Return ONLY valid JSON with this schema:

    {{
      "present": true or false or null,
      "lactate_value": number or null,
      "evidence_quote": string,
      "confidence": "low" or "medium" or "high"
    }}

    Note chunk:
    {note_chunk}
    """).strip()

    return f"[INST] {user_prompt} [/INST]"
