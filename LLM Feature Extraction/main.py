import pandas as pd
from datetime import datetime

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
) -> tuple[dict, list[str]]:
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
        return result, []

    chunk_results = []

    for chunk in chunks:
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
            chunk_results.append(parsed)

        except Exception as e:
            print(f"[WARN] Failed chunk: {e}")

            fallback = {k: None for k in json_keys}
            fallback["evidence_quote"] = ""
            fallback["confidence"] = "low"
            chunk_results.append(fallback)

    final_result = aggregator(chunk_results)
    final_result["chunk_results"] = chunk_results

    return final_result, chunks


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

    for index, row in df.iterrows():
        print(
            f"****************************************** "
            f"{feature_name} | index: {index} "
            f"******************************************"
        )

        result, chunks = run_feature_extraction_for_note(
            note_text=row["text"],
            keywords=feature_cfg["keywords"],
            prompt_builder=feature_cfg["prompt_builder"],
            json_keys=feature_cfg["json_keys"],
            aggregator=feature_cfg["aggregator"],
            empty_result=feature_cfg["empty_result"],
        )

        result_row = feature_cfg["result_row_builder"](row, result, chunks)
        results.append(result_row)

    out_df = pd.DataFrame(results)
    out_path = OUTPUT_DIR / (
        f'{feature_cfg["output_file_prefix"]}_'
        f'{now.strftime("%Y-%m-%d_%H-%M-%S")}.csv'
    )
    out_df.to_csv(out_path, index=False)

    print(f"Saved results to: {out_path}")


if __name__ == "__main__":
    main("shock") #main("shock") #main("lactate")