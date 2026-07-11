"""
Thin client for talking to the local llama.cpp server.

The extraction runs against a locally hosted model exposed through the
llama.cpp OpenAI-compatible chat endpoint, so no note text ever leaves the
machine. This module has a single helper that sends one prompt and returns the
raw model text.
"""

import requests


def call_llama_server(
    prompt: str,
    server_url: str,
    max_tokens: int = 80,
    temperature: float = 0.0
) -> str:
    """Send a prompt to the local llama server and return the model's reply.

    Wraps the prompt in a chat request with a short system message that asks the
    model to answer with valid JSON only. Raises a RuntimeError if the request
    times out or fails, so the caller can handle a bad chunk without crashing
    the whole run.
    """
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