# F1 — How LLMs work

## The one idea
An LLM does one thing: **given some text, predict the next token.** It repeats that, one token
at a time, until it decides to stop. Everything else (chat, tools, agents, RAG) is built on top.

## 1. Tokens
- Models don't read words or letters. Text is split into **tokens**: common chunks of characters.
  `"database"` may be 1 token, `"AWR_SNAPSHOT_ID"` may be 5.
- Rough rule for English: **1 token ≈ 4 characters ≈ ¾ of a word.**
- Why you care:
  - **Cost** — APIs charge per token (input and output priced separately).
  - **Limits** — the context window is measured in tokens.
  - **Languages** — Telugu/Hindi text takes many more tokens than the same sentence in English.
  - **Weird failures** — "how many r's in strawberry" is hard because the model sees tokens, not letters.

## 2. Next-token prediction and temperature
- At each step the model gives a **probability for every token in its vocabulary** (~150k for Qwen).
- **Sampling** picks one. **Temperature** reshapes the probabilities first:
  - `0` → always take the most likely token → same answer every time (deterministic-ish).
  - `~0.7` → some variety, still sensible.
  - `>1.2` → flattens the odds, unlikely tokens get picked → creative, then nonsense.
- **Rule of thumb:** extraction, classification, SQL, RCA → low temperature. Brainstorming → higher.

## 3. Context window
- The model's working memory: **prompt + conversation + output, all in tokens.**
- It has **no memory between calls.** A "chat" works because the app re-sends the whole history every time.
- Past the limit, text is cut off, so the model never sees it. Long contexts also cost more and get slower,
  and models attend less reliably to the middle of very long inputs.
- This is *why RAG exists*: you can't paste 10,000 invoices into the prompt, so you retrieve the 5 relevant ones.

## 4. Hallucination
- The model is trained to produce **plausible** text, not **true** text. When it doesn't know, the most
  plausible continuation is often a confident, made-up answer.
- Reduced (never removed) by:
  1. **Grounding** — give it the facts in the prompt and tell it to answer only from them (→ RAG).
  2. **Permission to not know** — "If the answer isn't in the context, say you don't know."
  3. **Verification** — check the output before acting on it (your interview narrative).

## Exercises (run from the repo root)
```bash
uv run python lessons/F1-how-llms-work/ex1_tokens.py
uv run python lessons/F1-how-llms-work/ex2_temperature.py
uv run python lessons/F1-how-llms-work/ex3_context_window.py
uv run python lessons/F1-how-llms-work/ex4_hallucination.py
```

## Interview questions you should now answer in 2–3 sentences
1. What is a token and why does it matter for cost and latency?
2. What does temperature do? What would you set it to for invoice field extraction, and why?
3. A chatbot "forgets" the start of a long conversation. Why?
4. Why do LLMs hallucinate, and what are three ways to reduce it in a production system?

## My results
_Fill in after running the exercises: one line per exercise on what surprised you._
