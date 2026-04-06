import pandas as pd
from datetime import datetime
import json

from config import (
    OUTPUT_DIR,
    FEATURE_CONFIGS,
    LLAMA_SERVER_URL,
    CHUNK_SENTENCE_SIZE,
    CHUNK_SENTENCE_OVERLAP,
    MAX_TOKENS,
    TEMPERATURE,
)
from note_processing import prepare_chunks
from llm_client import call_llama_server
from parsing import parse_llm_json


def run_feature_extraction_for_note(
    note_text: str,
    keywords: list[str],
    prompt_builder,
    json_keys: list[str],
    aggregator,
    empty_result: dict,
) -> tuple[dict, list[str], list[dict]]:
    chunks = prepare_chunks(
        note_text=note_text,
        keywords=keywords,
        chunk_size=CHUNK_SENTENCE_SIZE,
        overlap=CHUNK_SENTENCE_OVERLAP,
        window=1
    )

    if not chunks:
        result = empty_result.copy()
        result["chunk_results"] = []
        return result, [], []

    chunk_results = []
    chunk_debug_rows = []

    for i, chunk in enumerate(chunks):
        prompt = prompt_builder(chunk)

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

            parsed = parse_llm_json(raw_output, json_keys)

        except Exception as e:
            print(f"[WARN] Failed chunk {i}: {e}")

            raw_output = f"[ERROR] {e}"
            parsed = {k: None for k in json_keys}
            parsed["evidence_quote"] = ""
            parsed["confidence"] = "low"

        chunk_results.append(parsed)
        chunk_debug_rows.append({
            "chunk_index": i,
            "chunk_text": chunk,
            "raw_output": raw_output,
            "parsed_output": parsed,
        })

    final_result = aggregator(chunk_results)
    final_result["chunk_results"] = chunk_results

    return final_result, chunks, chunk_debug_rows


def main(feature_name: str = "lactate"):
    if feature_name not in FEATURE_CONFIGS:
        raise ValueError(
            f"Unknown feature: {feature_name}. "
            f"Available features: {list(FEATURE_CONFIGS.keys())}"
        )

    feature_cfg = FEATURE_CONFIGS[feature_name]
    now = datetime.now()

    df = pd.read_csv("../Outputs/discharge_filtered.csv")
    df_final_structured = pd.read_csv("../Outputs/final_structured_dataset.csv")
    df = df.merge(df_final_structured, how="inner", on=["hadm_id", "subject_id"])

    results = []
    chunk_level_results = []

    for index, row in df.iterrows():
        print(
            f"****************************************** "
            f"{feature_name} | index: {index} "
            f"******************************************"
        )

        result, chunks, chunk_debug_rows = run_feature_extraction_for_note(
            note_text=row["text"],
            keywords=feature_cfg["keywords"],
            prompt_builder=feature_cfg["prompt_builder"],
            json_keys=feature_cfg["json_keys"],
            aggregator=feature_cfg["aggregator"],
            empty_result=feature_cfg["empty_result"],
        )

        result_row = feature_cfg["result_row_builder"](row, result, chunks)

        # optional: keep compact JSON string in patient-level file
        result_row["chunk_debug_rows"] = json.dumps(chunk_debug_rows, ensure_ascii=False)
        results.append(result_row)

        # save chunk-level rows separately
        for debug_row in chunk_debug_rows:
            chunk_row = {
                "subject_id": row["subject_id"],
                "hadm_id": row["hadm_id"],
                "feature_name": feature_name,
                "chunk_index": debug_row["chunk_index"],
                "chunk_text": debug_row["chunk_text"],
                "raw_output": debug_row["raw_output"],
                "parsed_output": json.dumps(debug_row["parsed_output"], ensure_ascii=False),
            }
            chunk_level_results.append(chunk_row)

    out_df = pd.DataFrame(results)
    chunk_df = pd.DataFrame(chunk_level_results)

    timestamp = now.strftime("%Y-%m-%d_%H-%M-%S")

    out_path = OUTPUT_DIR / f'{feature_cfg["output_file_prefix"]}_{timestamp}.csv'
    chunk_out_path = OUTPUT_DIR / f'{feature_cfg["output_file_prefix"]}_chunks_{timestamp}.csv'

    out_df.to_csv(out_path, index=False)
    chunk_df.to_csv(chunk_out_path, index=False)

    print(f"Saved patient-level results to: {out_path}")
    print(f"Saved chunk-level results to: {chunk_out_path}")


if __name__ == "__main__":
    main("coma")  #main("lactate") #OR #main("shock") # main("coma")