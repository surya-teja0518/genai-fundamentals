"""Ex 3: The model only sees what fits in its context window."""
from common import ask

secret = "The on-call DBA this week is Priya, extension 4471."
filler = " ".join(
    f"Log line {i}: AWR snapshot {1000 + i} captured, no anomalies, avg active sessions 2.{i % 10}."
    for i in range(120)
)
question = "Who is the on-call DBA this week? Answer in one short line."
prompt = f"{secret}\n\n{filler}\n\n{question}"

for num_ctx in [4096, 512]:
    r = ask(prompt, temperature=0, num_ctx=num_ctx)
    print(f"context window = {num_ctx} tokens  (tokens the model actually read: {r['in_tokens']})")
    print(f"   answer: {r['text']}\n")

print("The fact sits at the START of the prompt. With a small window the start gets cut off,")
print("so the model never sees it — and may guess anyway (see Ex 4).")
