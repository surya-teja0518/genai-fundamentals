"""Day 3: first Claude API call — plain, streamed, with usage + cost tracking."""
import os
import sys

import anthropic
from dotenv import load_dotenv

load_dotenv()

MODEL = "claude-sonnet-5-5"
# USD per million tokens (input, output) — verify against current pricing page.
PRICE = {"claude-sonnet-5-5": (3.0, 15.0)}


def cost(model: str, usage) -> float:
    pin, pout = PRICE[model]
    return (usage.input_tokens * pin + usage.output_tokens * pout) / 1_000_000


def main() -> None:
    if not os.getenv("ANTHROPIC_API_KEY"):
        sys.exit("ANTHROPIC_API_KEY missing — copy .env.example to .env and fill it in.")

    client = anthropic.Anthropic()

    # 1. Plain call with a system prompt
    msg = client.messages.create(
        model=MODEL,
        max_tokens=300,
        system="You are a senior Oracle DBA. Be terse and concrete.",
        messages=[{"role": "user", "content": "Top 3 AWR sections to read first when a DB is slow, and why?"}],
    )
    print(msg.content[0].text)
    print(f"\n[usage] in={msg.usage.input_tokens} out={msg.usage.output_tokens} "
          f"stop={msg.stop_reason} cost=${cost(MODEL, msg.usage):.5f}\n")

    # 2. Streaming
    print("--- streaming ---")
    with client.messages.stream(
        model=MODEL,
        max_tokens=200,
        messages=[{"role": "user", "content": "In 3 bullets: what is hybrid (BM25 + dense) retrieval good for?"}],
    ) as stream:
        for text in stream.text_stream:
            print(text, end="", flush=True)
        final = stream.get_final_message()
    print(f"\n\n[usage] in={final.usage.input_tokens} out={final.usage.output_tokens} "
          f"cost=${cost(MODEL, final.usage):.5f}")


if __name__ == "__main__":
    main()
