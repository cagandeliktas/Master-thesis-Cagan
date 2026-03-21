import textwrap


def build_lactate_prompt(note_chunk: str) -> str:
    user_prompt = textwrap.dedent(f"""
    Extract whether lactate is elevated from the note chunk.

    Rules:
    - present=true if lactate > 2.0 mmol/L or explicitly described as elevated.
    - present=false if lactate <= 2.0 mmol/L or explicitly described as normal/cleared.
    - present=null if lactate is not mentioned.
    - Extract lactate_value if a numeric value is stated.
    - units should be "mmol/L" if stated or implied.

    Return ONLY valid JSON with this schema:

    {{
      "present": true or false or null,
      "lactate_value": number or null,
      "units": "mmol/L" or null,
      "evidence_quote": string,
      "confidence": "low" or "medium" or "high"
    }}

    Note chunk:
    {note_chunk}
    """).strip()

    return f"[INST] {user_prompt} [/INST]"


def build_shock_prompt_old(note_chunk: str) -> str:
    user_prompt = textwrap.dedent(f"""
    Extract whether shock is clinically documented in the note chunk.

    Rules:
    - present=true only if shock or circulatory failure is explicitly documented,
      or if the wording clearly indicates shock (e.g. cardiogenic shock, septic shock,
      refractory hemodynamic instability).
    - present=false only if the note explicitly states no shock, hemodynamically stable,
      or shock resolved.
    - present=null if shock is not clearly documented.
    - severity can be one of: "none", "mild", "moderate", "severe", or null.
    - Do not label shock as present based only on isolated lab abnormalities
      or a single low blood pressure reading.

    Return ONLY valid JSON with this schema:

    {{
      "present": true or false or null,
      "severity": "none" or "mild" or "moderate" or "severe" or null,
      "evidence_quote": string,
      "confidence": "low" or "medium" or "high"
    }}

    Note chunk:
    {note_chunk}
    """).strip()

    return f"[INST] {user_prompt} [/INST]"


def build_shock_prompt(note_chunk: str) -> str:
    user_prompt = textwrap.dedent(f"""
    Extract whether shock or clinically meaningful hemodynamic instability is documented in the note chunk.

    Rules:
    - present=true if shock is explicitly documented, or if the text clearly indicates clinically significant circulatory failure,
      such as cardiogenic shock, septic shock, persistent hypotension, vasopressor requirement,
      pressor-dependent hemodynamic support, or refractory hemodynamic instability.
    - present=false only if the note explicitly states no shock, hemodynamically stable, normotensive,
      off pressors with stability, or shock resolved.
    - present=null if shock/hemodynamic instability is not clearly documented.
    - severity can be one of: "none", "mild", "moderate", "severe", or null.
    - Do not label shock as present based only on isolated lab abnormalities
      or a single transient low blood pressure reading.

    Return ONLY valid JSON with this schema:

    {{
      "present": true or false or null,
      "severity": "none" or "mild" or "moderate" or "severe" or null,
      "evidence_quote": string,
      "confidence": "low" or "medium" or "high"
    }}

    Note chunk:
    {note_chunk}
    """).strip()

    return f"[INST] {user_prompt} [/INST]"