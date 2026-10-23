"""Ex 2: What temperature does. Part A is the math, part B is the real model."""
import numpy as np
from common import ask

# Part A: toy next-token probabilities after "The DB is slow because of high ..."
tokens = ["CPU", "IO", "waits", "latency", "bananas"]
logits = np.array([3.0, 2.5, 2.0, 1.0, -1.0])  # raw scores from the model

print("Part A — probability of each next token at different temperatures")
print(f"{'temp':>6} " + " ".join(f"{t:>9}" for t in tokens))
for temp in [0.1, 0.7, 1.0, 2.0]:
    p = np.exp(logits / temp)
    p /= p.sum()
    print(f"{temp:>6} " + " ".join(f"{x:>9.1%}" for x in p))

# Part B: same prompt, 4 runs per temperature
PROMPT = "Give a one-line name for a tool that finds slow SQL in Oracle. Reply with the name only."
print("\nPart B — same prompt, 4 runs each")
for temp in [0.0, 0.8, 1.8]:
    outs = [ask(PROMPT, temperature=temp)["text"] for _ in range(4)]
    print(f"\ntemperature={temp}  ({len(set(outs))} distinct answers out of 4)")
    for o in outs:
        print("   ", o[:90])
