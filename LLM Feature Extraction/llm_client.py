import requests


def call_llama_server(
    prompt: str,
    server_url: str,
    max_tokens: int = 80,
    temperature: float = 0.0
) -> str:
    url = f"{server_url}/v1/chat/completions"

    payload = {
        "model": "local-model",
        "messages": [
            {
                "role": "system",
                "content": (
                    "You are a precise clinical information extraction assistant. "
                    "Always return only valid JSON and nothing else."
                )
            },
            {
                "role": "user",
                "content": prompt
            }
        ],
        "temperature": temperature,
        "max_tokens": max_tokens
    }

    try:
        response = requests.post(url, json=payload, timeout=60)
        response.raise_for_status()
    except requests.Timeout:
        raise RuntimeError("LLM request timed out")
    except requests.RequestException as e:
        raise RuntimeError(f"LLM request failed: {e}")

    data = response.json()
    return data["choices"][0]["message"]["content"]