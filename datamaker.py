import json
import random

filename = "./data/naturalproofs_proofwiki.json"  # Update path if necessary

with open(filename, "r", encoding="utf-8") as f:
    data = json.load(f)

theorems = []
for item in data.get("dataset", {}).get("theorems", []):
    if "contents" in item and item["contents"]:
        # Join the array into one string (preserves LaTeX and text)
        full = " ".join(item["contents"]).strip()
    else:
        # Fallback to title/label if contents empty
        full = item.get("title", item.get("label", "")).strip()

    # Clean up extra whitespace and skip very short/empty
    full = " ".join(full.split())
    if full and len(full) > 30:  # Adjustable threshold for meaningful theorems
        theorems.append(full)

print(f"Extracted {len(theorems)} theorem statements!")

# Grab 200 for our subset (there are thousands total, so plenty)
num_to_use = 200
subset = random.sample(theorems, min(num_to_use, len(theorems)))

# Save
with open("math_truths_subset.txt", "w", encoding="utf-8") as out:
    for t in subset:
        out.write(t + "\n\n===\n\n")

print(f"Saved {len(subset)} theorems to math_truths_subset.txt")
print("\nFirst 5 examples from subset:")
for t in subset[:5]:
    print(t)
    print("---")
