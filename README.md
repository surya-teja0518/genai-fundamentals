# genai-fundamentals

Gen AI engineering fundamentals: one folder of notes and a small runnable exercise per module.

| ID | Module | Status |
|---|---|---|
| F1 | How LLMs work: tokens, context, sampling | next |
| F2 | Claude API basics | in progress (`scripts/01_first_call.py`) |
| F3 | Prompt engineering | |
| F4 | Structured outputs | |
| F5 | Tool use and the agent loop | |
| F6 | Embeddings and similarity | in progress (`scripts/02_embeddings.py`) |
| F7 | Chunking and vector databases | |
| F8 | Better retrieval: BM25, hybrid, reranking | |
| F9 | RAG end to end | |
| F10 | Evaluation | |
| F11 | LangGraph | |
| F12 | MCP | |
| F13 | Observability | |

## Run
```bash
cp .env.example .env   # add ANTHROPIC_API_KEY
uv run python scripts/01_first_call.py
uv run python scripts/02_embeddings.py
```

## Findings
- **F6:** dense-only retrieval ranked an AWR note about the ORDERS *index* above the goods-receipt
  note for "quantity mismatch between order and delivery". The model matched the word "order",
  not the meaning. This is the reason for BM25 + hybrid search + reranking in F8.
