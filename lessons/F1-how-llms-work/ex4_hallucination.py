"""Ex 4: Make the model hallucinate, then reduce it with grounding + permission to say 'I don't know'."""
from common import ask

# This parameter does not exist.
Q = "What does the Oracle 19c initialization parameter _awr_turbo_snapshot do, and what is its default value?"

print("1) Plain question")
print("  ", ask(Q, temperature=0)["text"][:600], "\n")

print("2) Same question + permission to not know")
sys = "You are an Oracle expert. If you are not certain a parameter exists, say 'I don't know' instead of guessing."
print("  ", ask(Q, temperature=0, system=sys)["text"][:600], "\n")

print("3) Grounded: answer only from supplied documentation")
docs = (
    "Documentation excerpt:\n"
    "- _awr_snapshot_time_offset: offsets AWR snapshot times to spread load. Default 0.\n"
    "- awr_pdb_autoflush_enabled: enables AWR snapshots in PDBs. Default FALSE.\n"
)
sys = "Answer ONLY from the documentation provided. If it is not there, reply exactly: Not in the provided documentation."
print("  ", ask(f"{docs}\nQuestion: {Q}", temperature=0, system=sys)["text"][:600])

print("\nNote: a small 3B model hallucinates a lot. Bigger models do it less, but never zero —")
print("that's why production systems ground answers and verify them.")
