"""Shared helper: call the local Ollama model."""
import ollama

MODEL = "qwen2.5:3b"


def ask(prompt: str, temperature: float = 0.7, num_ctx: int = 4096, system: str | None = None) -> dict:
    messages = ([{"role": "system", "content": system}] if system else []) + [{"role": "user", "content": prompt}]
    r = ollama.chat(model=MODEL, messages=messages, options={"temperature": temperature, "num_ctx": num_ctx})
    return {
        "text": r.message.content.strip(),
        "in_tokens": r.prompt_eval_count,
        "out_tokens": r.eval_count,
    }
