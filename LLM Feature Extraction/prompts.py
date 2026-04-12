import textwrap


def build_lactate_prompt(note_chunk: str) -> str:
    user_prompt = textwrap.dedent(f"""
    Extract whether lactate is elevated in this note chunk.

    Definitions:
    - Elevated lactate means lactate > 2.0 mmol/L, or explicit wording such as
      "elevated lactate" or "lactic acidosis".
    - Not elevated means lactate <= 2.0 mmol/L, or explicit wording such as
      "normal lactate", "lactate cleared", or "lactate normalized".

    Rules:
    - present=true if the chunk contains evidence of elevated lactate.
    - present=false if the chunk contains evidence that lactate is not elevated.
    - present=null if lactate is not mentioned clearly enough.
    - If a numeric lactate value is stated, extract lactate_value.
    - If multiple lactate values appear in the chunk, extract the highest clearly stated value.
    - units should be "mmol/L" if stated or clearly implied.
    - Do not guess a value if no numeric value is written.
    - Prefer exact wording from the chunk for evidence_quote.

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
    Extract whether shock or clinically meaningful hemodynamic instability is documented in this note chunk.

    Definitions:
    - Shock / hemodynamic instability includes explicit documentation of shock
      (e.g. cardiogenic shock, septic shock, hemorrhagic shock),
      circulatory failure, refractory hypotension, vasopressor-dependent hypotension,
      or persistent pressor requirement for hemodynamic support.

    Task framing:
    - This is a retrospective documentation task.
    - Label present=true if the chunk documents that shock or clinically meaningful
      hemodynamic instability occurred at any point, even if the chunk later says
      the shock improved or resolved.
    - Label present=false if the chunk does not contain sufficient evidence that shock
      or clinically meaningful hemodynamic instability was documented.

    Rules:
    - present=true if the chunk explicitly documents shock, circulatory failure,
      refractory or persistent hypotension, vasopressor-dependent instability,
      or another clearly stated hemodynamic instability state.
    - present=true also if the chunk describes prior shock/hemodynamic instability
      that later improved or resolved.
    - present=false for all other cases, including:
      - explicit stability or absence of shock (e.g. hemodynamically stable, no shock,
        normotensive, off pressors with stability)
      - no mention of shock/hemodynamic instability
      - ambiguous or weak evidence that does not clearly document shock
    - Do not label shock as present based only on isolated laboratory abnormalities,
      a single low blood pressure reading, vague illness severity, PEA arrest alone,
      multiorgan failure alone, or shock liver alone.
    - Do not infer shock solely from the fact that vasopressors were mentioned unless the
      text indicates they were being used for hemodynamic support or hypotension.

    Severity:
    - severity can be one of: "none", "mild", "moderate", "severe".
    - If present=true, assign severity based on the strongest active shock /
      hemodynamic instability documented anywhere in the chunk, even if the chunk
      later states that shock improved or resolved.
    - Use "severe" for wording such as refractory shock, profound shock, multiple pressors,
      pressor escalation, or severe hemodynamic instability.
    - Use "moderate" for clear shock/hemodynamic instability without severe descriptors.
    - Use "mild" only for limited or improving instability.
    - Use "none" when present=false.

    Evidence:
    - Prefer exact wording from the chunk for evidence_quote.
    - If present=true, quote the phrase that best supports shock/hemodynamic instability.
    - If present=false and there is an explicit negative phrase, quote it.
    - If present=false because there is simply no clear positive evidence, evidence_quote may be empty.

    Return ONLY valid JSON with this schema:
    {{
      "present": true or false,
      "severity": "none" or "mild" or "moderate" or "severe",
      "evidence_quote": string,
      "confidence": "low" or "medium" or "high"
    }}

    Note chunk:
    {note_chunk}
    """).strip()

    return f"[INST] {user_prompt} [/INST]"


def build_shock_prompt(note_chunk: str) -> str:
    user_prompt = textwrap.dedent(f"""
    You are analyzing a clinical discharge note to determine whether the patient experienced
    shock or clinically meaningful hemodynamic instability at any point during the hospital stay.

    Use retrospective labeling:
    - If instability occurred at any time, classify it as present, even if it later resolved.

    Step 1: Identify evidence
    Determine whether the note contains any of the following:
    - explicit mention of shock
    - vasopressor use documented (e.g. norepinephrine, epinephrine)
    - hypotension or hemodynamic instability described
    - evidence of escalation or refractory instability (e.g. multiple pressors, refractory instability)

    Step 2: Apply rules
    - Vasopressor use in a clinical context suggests hemodynamic instability unless clearly stated otherwise.
    - Do not rely on isolated blood pressure readings or lab values alone.

    Step 3: Assign severity
    - "severe": refractory or escalating instability, or multiple vasopressors
    - "moderate": vasopressor-dependent instability
    - "mild": instability without vasopressors
    - "none": no instability

    Step 4: Output JSON
    - If shock_present=true, shock_severity must be "mild", "moderate", or "severe".
    - If shock_present=false, shock_severity must be "none".
    - If shock_present=null, shock_severity must be null.
    - evidence_quotes should contain the most relevant exact supporting quote(s) from the note.
    - justification should briefly explain why the evidence supports the final label.
    - confidence should be a number between 0.0 and 1.0.

    Return ONLY valid JSON with this schema:
    {{
      "shock_present": true or false or null,
      "shock_severity": "none" or "mild" or "moderate" or "severe" or null,
      "evidence_quotes": ["string"],
      "justification": "string",
      "confidence": 0.0
    }}

    Note chunk:
    {note_chunk}
    """).strip()

    return f"[INST] {user_prompt} [/INST]"


def build_coma_prompt(note_chunk: str) -> str:
    user_prompt = textwrap.dedent(f"""
    Extract whether coma or unresponsiveness is documented in this note chunk.

    Definitions:
    - Coma / unresponsive documentation refers to clear documentation of severely impaired
      consciousness, such as "comatose", "unresponsive", "no response", "no purposeful response",
      "not responsive to stimuli", or equivalent wording.
    - This feature is intended to capture neurological injury severity after ROSC.
    - This should reflect neurological unresponsiveness, not sedation alone.

    Task framing:
    - This is a retrospective documentation task.
    - Label present=true if coma or unresponsiveness is documented at any point,
      even if the patient later improved or regained consciousness.
    - Label present=false if there is no clear evidence of coma or unresponsiveness.

    Rules:
    - present=true if the chunk explicitly documents coma, unresponsiveness, or another clearly
      stated severely depressed level of consciousness.
    - present=true also if the chunk describes prior coma/unresponsiveness that later improved.
    - present=false for all other cases, including:
      - explicit normal/adequate responsiveness (e.g., awake, alert, following commands)
      - no mention of coma/unresponsiveness
      - ambiguous or weak evidence that does not clearly document coma/unresponsiveness

    Important exclusions:
    - Do NOT label present=true based only on:
      - sedation (e.g., "sedated", "on propofol", "intubated and sedated")
      - intubation or mechanical ventilation alone
      - vague phrases such as "altered mental status", "encephalopathy", "lethargic",
        "somnolent", or "confused" without clear unresponsiveness
      - low GCS alone unless the text clearly supports coma/unresponsiveness

    Sedation handling:
    - If sedation is mentioned, only label present=true if there is clear evidence of
      neurological unresponsiveness beyond sedation.
    - If both sedation and unresponsiveness are mentioned, prioritize explicit neurological
      descriptions (e.g., "remains unresponsive off sedation").

    Evidence:
    - Prefer exact wording from the chunk for evidence_quote.
    - If present=true, quote the phrase that best supports coma/unresponsiveness.
    - If present=false and there is an explicit normal/responsive phrase, quote it.
    - If present=false due to absence of clear evidence, evidence_quote may be empty.

    Return ONLY valid JSON with this schema:
    {{
      "present": true or false,
      "evidence_quote": string,
      "confidence": "low" or "medium" or "high"
    }}

    Note chunk:
    {note_chunk}
    """).strip()

    return f"[INST] {user_prompt} [/INST]"