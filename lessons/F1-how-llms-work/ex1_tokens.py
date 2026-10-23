"""Ex 1: See how text becomes tokens, using Qwen's own tokenizer."""
from transformers import AutoTokenizer

tok = AutoTokenizer.from_pretrained("Qwen/Qwen2.5-3B-Instruct")
print(f"Vocabulary size: {tok.vocab_size:,} tokens\n")

SAMPLES = {
    "English": "The database is running slowly today.",
    "Telugu": "ఈరోజు డేటాబేస్ నెమ్మదిగా నడుస్తోంది.",
    "Hindi": "आज डेटाबेस धीरे चल रहा है।",
    "SQL": "SELECT sql_id, elapsed_time FROM v$sql ORDER BY 2 DESC FETCH FIRST 5 ROWS ONLY;",
    "Oracle wait": "enq: TX - row lock contention",
    "Invoice id": "INV-2041-ACME-2026",
    "Number": "12400.75",
    "strawberry": "strawberry",
}

for name, text in SAMPLES.items():
    ids = tok.encode(text)
    pieces = [tok.decode([i]) for i in ids]
    print(f"{name:12} chars={len(text):3}  tokens={len(ids):3}  chars/token={len(text)/len(ids):.1f}")
    print(f"{'':12} {pieces}\n")

print("Questions: Which language costs the most tokens for the same meaning?")
print("How is 'strawberry' split, and why does that make counting its r's hard?")
