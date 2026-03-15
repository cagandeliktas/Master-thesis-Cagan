import pandas as pd

from config import (
    LLAMA_SERVER_URL,
    LACTATE_KEYWORDS,
    CHUNK_SENTENCE_SIZE,
    CHUNK_SENTENCE_OVERLAP,
    MAX_TOKENS,
    TEMPERATURE,
    OUTPUT_DIR,
)
from prompts import build_lactate_prompt
from note_processing import prepare_lactate_chunks
from llm_client import call_llama_server
from parsing import parse_lactate_json
from aggregation import aggregate_lactate_chunk_results
from datetime import datetime

now = datetime.now() # get the time


def run_lactate_extraction_for_note(note_text: str) -> tuple[dict, list[str]]:
    chunks = prepare_lactate_chunks(
        note_text=note_text,
        keywords=LACTATE_KEYWORDS,
        chunk_size=CHUNK_SENTENCE_SIZE,
        overlap=CHUNK_SENTENCE_OVERLAP,
        window=1
    )

    if not chunks:
        return {
            "final_present": None,
            "final_lactate_value": None,
            "final_units": None,
            "final_evidence_quote": "",
            "n_chunks": 0,
            "chunk_results": []
        }, []

    chunk_results = []

    for chunk in chunks:
        prompt = build_lactate_prompt(chunk)
        try:
            raw_output = call_llama_server(
                prompt=prompt,
                server_url=LLAMA_SERVER_URL,
                max_tokens=MAX_TOKENS,
                temperature=TEMPERATURE,
            )

            print("raw output")
            print(raw_output)
            print("**********************************************************************")

            parsed = parse_lactate_json(raw_output)
            chunk_results.append(parsed)

        except Exception as e:
            print(f"[WARN] Failed chunk: {e}")
            chunk_results.append({
                "present": None,
                "lactate_value": None,
                "units": None,
                "evidence_quote": "",
                "confidence": "low"
            })

    final_result = aggregate_lactate_chunk_results(chunk_results)
    final_result["chunk_results"] = chunk_results

    return final_result, chunks


def main():
    df = pd.read_csv("../Outputs/discharge_filtered.csv")
    df_final_strucuted = pd.read_csv("../Outputs/final_structured_dataset.csv")
    df = df.merge(df_final_strucuted, how='inner', on=['hadm_id', 'subject_id'])


    results = []

    for index, row in df.iterrows():
        print(f' ****************************************** index: {index} ******************************************')
        result, chunks = run_lactate_extraction_for_note(row["text"])

        structured_label = None
        if pd.notna(row.get("structured_lactate_max")):
            structured_label = bool(row["structured_lactate_max"] > 2)

        results.append({
            'note_id': row['note_id'],
            "hadm_id": row["hadm_id"],
            'note_type': row['note_type'],
            "llm_present": result["final_present"],
            "llm_lactate_value": result["final_lactate_value"],
            "llm_evidence_quote": result["final_evidence_quote"],
            "n_chunks": result["n_chunks"],
            'chunks': chunks,
            "structured_lactate": row.get("Lactate"),
        })
    print('Results: ')
    print(results)

    out_df = pd.DataFrame(results)
    out_path = OUTPUT_DIR / f"lactate_extraction_results_{now.strftime("%Y-%m-%d %H:%M:%S")}.csv"
    out_df.to_csv(out_path, index=False)

    print(f"Saved results to: {out_path}")


if __name__ == "__main__":
    main()