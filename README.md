# Veraciscope

Experimental tool for exploring truth geometry in embedding space.

The idea: embed known mathematical truths, falsehoods, and edge cases (Goldbach conjecture, Riemann hypothesis, Russell's paradox, etc.) into vector space. Cluster the truth embeddings, build polygonal hulls from the clusters, and look for what falls in the "shadow" -- undiscovered connections or ambiguous statements that sit near the truth boundary.

## Status

Early prototype (January 2026). Four iterations exploring different clustering, hull construction, and visualization approaches. Produces 2D UMAP plots and interactive 3D polyhedron visualizations.

## Data

- `data/math_truths_subset.txt` -- Curated subset of mathematical theorems (included)
- `data/naturalproofs_proofwiki.json` -- Full ProofWiki dataset from [NaturalProofs](https://github.com/wellecks/naturalproofs) (~116 MB, not included -- download separately if needed)

## Requirements

- Python 3.11+
- ollama (running locally with `qwen3-embedding:8b` model)
- numpy, scikit-learn, umap-learn, matplotlib, scipy
