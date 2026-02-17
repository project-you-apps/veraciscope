import ollama
import numpy as np
from sklearn.metrics.pairwise import cosine_similarity
import umap
import matplotlib.pyplot as plt
import os

with open("./data/math_truths_subset.txt", "r", encoding="utf-8") as f:
    math_theorems = [line.strip() for line in f.read().split("===") if line.strip()]

truths = math_theorems  # ~ 200 real theorems from ProofWiki, extracted by datamaker.py

print(
    f"Loaded {len(truths)} truth statements (including {len(math_theorems)} math theorems)"
)

# Ollama embedding models (pull these first!)
model_names = [
    # "all-minilm",  # Fast & lightweight
    "nomic-embed-text",  # Excellent quality, long context
    # "mxbai-embed-large",  # SOTA open embedding (if pulled)
    # "qwen2.5:32b-instruct-q4_K_M"
    "qwen3-embedding:8b",
]

"""
# Truths / Falsehoods / Tests (same as before)
truths = [
    "2 + 2 = 4",
    "The sky is blue on clear days",
    "Water is wet",
    "Paris is the capital of France",
    "The Earth orbits the Sun",
    "Apples fall downward due to gravity",
    "Water boils at 100 degrees Celsius at sea level",
    "The speed of light is approximately 299792 kilometers per second",
    "Humans have 46 chromosomes",
    "The atomic number of oxygen is 8",
]
"""

falsehoods = [
    "2 + 2 = 5",
    "The sky is green",
    "Water is dry",
    "Paris is the capital of Germany",
    "The Sun orbits the Earth",
    "Apples fall upward",
    "The Earth is flat",
    "Water boils at -10 degrees Celsius",
]

tests = [
    "The sky appears orange during sunset",
    "This statement is false",
    "Every even integer greater than 2 can be expressed as the sum of two primes",  # Goldbach
    "All non-trivial zeros of the Riemann zeta function have real part 1/2",  # Riemann
    "P vs NP has been solved",
    "Schrödinger's cat is both alive and dead until observed",
    "The present king of France is bald",
]


# Helper to get embeddings via Ollama (batched for speed)
def get_embeddings(texts, model_name):
    response = ollama.embeddings(
        model=model_name, prompt=texts
    )  # Note: some models use 'prompt', others 'input'
    return (
        np.array(response["embeddings"])
        if "embeddings" in response
        else np.array(response["embedding"])[None, :]
    )


# Some models (like nomic) expect "prompt" key, others "input"—ollama client handles it, but batch via list
def batch_embed(texts, model_name):
    return np.array(
        [ollama.embeddings(model=model_name, prompt=t)["embedding"] for t in texts]
    )


# Pre-compute centroids per model
centroids = []
truth_embeddings_per_model = []

for model_name in model_names:
    print(f"Embedding truths with {model_name}...")
    truth_emb = batch_embed(truths, model_name)
    truth_embeddings_per_model.append(truth_emb)
    centroids.append(np.mean(truth_emb, axis=0).reshape(1, -1))


def veraciscope_score(statement, use_mahalanobis=False):
    sims = []
    for i, model_name in enumerate(model_names):
        emb = batch_embed([statement], model_name)
        if use_mahalanobis:
            cov = (
                np.cov(truth_embeddings_per_model[i], rowvar=False)
                + np.eye(truth_embeddings_per_model[i].shape[1]) * 1e-6
            )
            inv_cov = np.linalg.inv(cov)
            dist = pairwise_distances(
                emb, centroids[i], metric="mahalanobis", VI=inv_cov
            )[0][0]
            score = 1 / (1 + dist)
        else:
            score = cosine_similarity(emb, centroids[i])[0][0]
        sims.append(score)
    return np.mean(sims), sims


# Ablation
print("\n=== Ablation ===")
truth_scores = [veraciscope_score(t)[0] for t in truths]
false_scores = [veraciscope_score(t)[0] for t in falsehoods]
print(f"Avg truth: {np.mean(truth_scores):.4f} (±{np.std(truth_scores):.4f})")
print(f"Avg false: {np.mean(false_scores):.4f} (±{np.std(false_scores):.4f})")
threshold = (np.mean(truth_scores) + np.mean(false_scores)) / 2
print(f"Threshold: {threshold:.4f}\n")

# Tests
print("=== Tests ===")
for stmt in tests:
    score, per_model = veraciscope_score(stmt)
    verdict = (
        "Truth-like"
        if score > threshold + 0.02
        else "False-like" if score < threshold - 0.02 else "Ambiguous"
    )
    print(f"{score:.4f} | {verdict} | {stmt}")
    print(f"   Per-model: {['%.3f' % s for s in per_model]}\n")

# Visualization (using nomic for plot)
print("Generating UMAP plot with nomic-embed-text...")
all_texts = truths + falsehoods + tests
all_emb = batch_embed(all_texts, "nomic-embed-text")
reducer = umap.UMAP(n_neighbors=5, min_dist=0.3, random_state=42)
proj = reducer.fit_transform(all_emb)

plt.figure(figsize=(12, 9))
colors = ["blue"] * len(truths) + ["red"] * len(falsehoods) + ["green"] * len(tests)
plt.scatter(proj[:, 0], proj[:, 1], c=colors, alpha=0.8)
for i, txt in enumerate(all_texts):
    plt.annotate(f"{i}", (proj[i, 0], proj[i, 1]), fontsize=8, alpha=0.7)
plt.title("Veraciscope UMAP - Blue=Truth, Red=False, Green=Test (nomic-embed-text)")
plt.show()
