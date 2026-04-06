from pathlib import Path

from prompts import build_lactate_prompt, build_shock_prompt, build_coma_prompt
from aggregation import (
    aggregate_lactate_chunk_results,
    aggregate_shock_chunk_results,
    aggregate_coma_chunk_results,
)
from result_builders import (
    build_lactate_result_row,
    build_shock_result_row,
    build_coma_result_row,
)


BASE_DIR = Path(__file__).resolve().parent
OUTPUT_DIR = BASE_DIR / "outputs"
OUTPUT_DIR.mkdir(exist_ok=True)

LLAMA_SERVER_URL = "http://127.0.0.1:8080"

CHUNK_SENTENCE_SIZE = 5
CHUNK_SENTENCE_OVERLAP = 1
MAX_TOKENS = 200
TEMPERATURE = 0.0


LACTATE_KEYWORDS = ["lactate", "lactic"]
LACTATE_KEYS = ["present", "lactate_value", "units", "evidence_quote", "confidence"]

SHOCK_KEYWORDS_STRICT = [
    "shock",
    "septic shock",
    "cardiogenic shock",
    "hemorrhagic shock",
    "distributive shock",
    "hemodynamic instability",
    "hemodynamically unstable",
    "circulatory failure",
    "persistent hypotension",
    "refractory hypotension",
    "hypotensive requiring",
    "pressor",
    "pressors",
    "vasopressor",
    "vasopressin",
    "levophed",
    "norepinephrine",
    "epinephrine gtt",
    "phenylephrine",
    "neo",
    "off pressors",
    "shock resolved"
]
SHOCK_KEYS = ["present", "severity", "evidence_quote", "confidence"]


EMPTY_LACTATE_RESULT = {
    "final_present": None,
    "final_lactate_value": None,
    "final_evidence_quote": "",
    "n_chunks": 0,
}

EMPTY_SHOCK_RESULT = {
    "final_present": None,
    "final_severity": None,
    "final_evidence_quote": "",
    "n_chunks": 0,
}

COMA_KEYWORDS = [
    "coma",
    "comatose",
    "unresponsive",
    "unresponsive to stimuli",
    "unresponsive off sedation",
    "remains unresponsive",
    "still unresponsive",
    "not responsive",
    "not responsive to stimuli",
    "non-responsive",
    "non responsive",
    "non-responsive to stimuli",
    "no response",
    "no purposeful response",
    "no purposeful movements",
    "no meaningful response",
    "does not respond",
    "does not respond to stimuli",
    "not arousable",
    "unarousable",
    "unconscious",
    "loss of consciousness",
    "lost consciousness",
    "minimally responsive",
    "poorly responsive",
    "responsive only to pain",
    "no spontaneous movement",
    "not following commands",
    "unable to follow commands",
    "does not follow commands",
    "obtunded",
    "stuporous",
    "decreased level of consciousness",
    "depressed level of consciousness",
    "off sedation",
    "sedation held",
    "sedation stopped",
    "sedation weaned",
    "not waking up",
    "failed to awaken",
    "failed to wake up",
    "awake",
    "alert",
    "following commands",
    "follows commands",
    "responsive",
    "arousable",
    "not following commands",
    "unable to follow commands",
    "not following commands off sedation"
]

COMA_KEYS = ["present", "evidence_quote", "confidence"]

EMPTY_COMA_RESULT = {
    "final_present": None,
    "final_evidence_quote": "",
    "n_chunks": 0,
}


FEATURE_CONFIGS = {
    "lactate": {
        "keywords": LACTATE_KEYWORDS,
        "json_keys": LACTATE_KEYS,
        "prompt_builder": build_lactate_prompt,
        "aggregator": aggregate_lactate_chunk_results,
        "empty_result": EMPTY_LACTATE_RESULT,
        "result_row_builder": build_lactate_result_row,
        "output_file_prefix": "lactate_extraction_results",
    },
    "shock": {
        "keywords": SHOCK_KEYWORDS_STRICT,
        "json_keys": SHOCK_KEYS,
        "prompt_builder": build_shock_prompt,
        "aggregator": aggregate_shock_chunk_results,
        "empty_result": EMPTY_SHOCK_RESULT,
        "result_row_builder": build_shock_result_row,
        "output_file_prefix": "shock_extraction_results",
    },
    "coma": {
        "keywords": COMA_KEYWORDS,
        "json_keys": COMA_KEYS,
        "prompt_builder": build_coma_prompt,
        "aggregator": aggregate_coma_chunk_results,
        "empty_result": EMPTY_COMA_RESULT,
        "result_row_builder": build_coma_result_row,
        "output_file_prefix": "coma_extraction_results",
    },
}