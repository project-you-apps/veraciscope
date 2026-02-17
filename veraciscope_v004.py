import ollama
import numpy as np
from sklearn.metrics.pairwise import (
    cosine_similarity,
    pairwise_distances,
)  # Fixed import here
import umap
import matplotlib.pyplot as plt
import os

# Single model
model_name = "qwen3-embedding:8b"

# Load math theorems
with open("./data/math_truths_subset.txt", "r", encoding="utf-8") as f:
    truths = [line.strip() for line in f.read().split("===") if line.strip()]

print(f"Loaded {len(truths)} math theorem truths")

# Falsehoods and tests (keep as-is)
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
    "Some elephants have big ears. Some elephants do not have big ears.",
    "There is a mouse named Mickey that has a friend named Minnie.",
    "The statement 'This statement is false' is a paradox.",
    "The Riemann Hypothesis is one of the most famous unsolved problems in mathematics.",
    "The P vs NP problem asks whether every problem whose solution can be quickly verified can also be quickly solved.",
    "Schrödinger's cat is a thought experiment that illustrates the concept of superposition in quantum mechanics.",
    "The present king of France is bald is a classic example of a statement that is neither true nor false, as there is no current king of France.",
    "A tomato is a fruit.",
]


# Updated batch_embed with prefixes and type distinction
def batch_embed(
    texts, model_name, embed_type="document", max_chars=8000
):  # qwen3 handles long inputs
    embeddings = []
    for i, t in enumerate(texts):
        truncated = t[:max_chars]
        if embed_type == "document":
            prompt = "Represent this mathematical theorem for retrieval: " + truncated
        else:  # query
            prompt = (
                "Instruct: Retrieve similar proven mathematical theorems.\nQuery: "
                + truncated
            )

        try:
            resp = ollama.embeddings(model=model_name, prompt=prompt)
            embeddings.append(resp["embedding"])
        except Exception as e:
            print(f"Warning: Failed on item {i} ({model_name}): {str(e)[:100]}")
            embeddings.append(
                [0.0] * 1024
            )  # Placeholder (dim may vary; check first run)
    return np.array(embeddings)


# Embed truths (as documents)
print(f"Embedding truths with {model_name}...")
truth_embeddings = batch_embed(truths, model_name, embed_type="document")
truth_centroid = np.mean(truth_embeddings, axis=0).reshape(1, -1)

# Embed falsehoods and tests (as queries for fairness, or 'document' if you prefer symmetric)
print("Embedding falsehoods and tests...")
false_embeddings = batch_embed(falsehoods, model_name, embed_type="query")
test_embeddings = batch_embed(tests, model_name, embed_type="query")

# Rest of scoring/visualization (same as before, but single-model simplified)
truth_sims = cosine_similarity(truth_embeddings, truth_centroid).diagonal()
false_sims = cosine_similarity(false_embeddings, truth_centroid).flatten()
test_sims = cosine_similarity(test_embeddings, truth_centroid).flatten()


# For Mahalanobis (now works with the import fix)
# Toggle use_mahalanobis=True in calls to test it
"""
def veraciscope_score(
    statement, use_mahalanobis=True
):  # Default to True for better separation
    emb = batch_embed([statement], model_name, embed_type="query")  # Query prefix
    if use_mahalanobis:
        cov = (
            np.cov(truth_embeddings, rowvar=False)
            + np.eye(truth_embeddings.shape[1]) * 1e-6
        )
        inv_cov = np.linalg.inv(cov)
        dist = pairwise_distances(
            emb, truth_centroid, metric="mahalanobis", VI=inv_cov
        )[0][0]
        score = 1 / (1 + dist)
    else:
        score = cosine_similarity(emb, truth_centroid)[0][0]
    return score  # Single value now
"""


def veraciscope_score(statement, use_mahalanobis=False):
    emb = batch_embed([statement], model_name, embed_type="query")
    if use_mahalanobis:
        cov = (
            np.cov(truth_embeddings, rowvar=False)
            + np.eye(truth_embeddings.shape[1]) * 1e-6
        )
        inv_cov = np.linalg.inv(cov)
        dist = pairwise_distances(
            emb, truth_centroid, metric="mahalanobis", VI=inv_cov
        )[0][0]
        score = 1 / (1 + dist)
    else:
        score = cosine_similarity(emb, truth_centroid)[0][0]
    return score  # Single float


"""
# Visualization (save to results, etc.)
# Ablation + tests printout...
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
"""

# Ablation results (updated for single-model)
print("\n=== Ablation ===")
truth_scores = [veraciscope_score(t) for t in truths]  # Just the score
false_scores = [veraciscope_score(f) for f in falsehoods]
test_scores = [veraciscope_score(stmt) for stmt in tests]  # For later if needed

print(
    f"Avg truth similarity: {np.mean(truth_scores):.4f} (±{np.std(truth_scores):.4f})"
)
print(
    f"Avg false similarity: {np.mean(false_scores):.4f} (±{np.std(false_scores):.4f})"
)
sep = np.mean(truth_scores) - np.mean(false_scores)
print(f"Separation: {sep:.4f}")
threshold = (np.mean(truth_scores) + np.mean(false_scores)) / 2
print(f"Suggested threshold: {threshold:.4f}\n")

# Test statements (updated loop)
print("=== Test Statements ===")
for stmt, sim in zip(tests, test_scores):
    verdict = (
        "Truth-like"
        if sim > threshold + 0.02
        else "False-like" if sim < threshold - 0.02 else "Ambiguous"
    )
    print(f"{sim:.4f} | {verdict} | {stmt}")

# Visualization (now using the actual model)
print(f"Generating UMAP plot with {model_name}...")
all_texts = truths + falsehoods + tests
all_emb = np.vstack(
    [truth_embeddings, false_embeddings, test_embeddings]
)  # Reuse already-computed qwen3 embeddings!

reducer = umap.UMAP(n_neighbors=10, min_dist=0.3, random_state=42)
proj = reducer.fit_transform(all_emb)

plt.figure(figsize=(12, 10))
colors = ["blue"] * len(truths) + ["red"] * len(falsehoods) + ["green"] * len(tests)
plt.scatter(proj[:, 0], proj[:, 1], c=colors, alpha=0.8)

# Annotate tests for clarity
for i, txt in enumerate(tests):
    idx = len(truths) + len(falsehoods) + i
    short_label = txt[:30].replace("\n", " ") + "..." if len(txt) > 30 else txt
    plt.annotate(short_label, (proj[idx, 0], proj[idx, 1]), fontsize=9, alpha=0.9)

plt.title(f"Veraciscope UMAP - Blue=Truth, Red=False, Green=Test ({model_name})")
plt.legend(["Truth (Theorems)", "False", "Test"])
os.makedirs("./results", exist_ok=True)
plt.savefig(
    f'./results/umap_{model_name.replace(":", "_")}_mahalanobis.png',
    dpi=150,
    bbox_inches="tight",
)
plt.show()

from scipy.spatial import ConvexHull
from matplotlib.path import Path

# After proj = reducer.fit_transform(all_emb)

# Compute convex hull on truth points only
truth_proj = proj[: len(truths)]
hull = ConvexHull(truth_proj)

# Plot the hull (faceted polygon)
hull_path = Path(truth_proj[hull.vertices])
patch = plt.Polygon(
    truth_proj[hull.vertices],
    closed=True,
    fill=True,
    alpha=0.2,
    color="blue",
    label="Truth Polygon (Convex Hull)",
)
plt.gca().add_patch(patch)

# Outline the facets
plt.plot(
    truth_proj[hull.vertices, 0], truth_proj[hull.vertices, 1], "b--", lw=1, alpha=0.7
)
plt.plot(
    np.append(truth_proj[hull.vertices, 0], truth_proj[hull.vertices[0], 0]),
    np.append(truth_proj[hull.vertices, 1], truth_proj[hull.vertices[0], 1]),
    "b--",
    lw=1,
)

# Classify test points (green)
test_proj = proj[-len(tests) :]
for i, point in enumerate(test_proj):
    inside = hull_path.contains_point(point)
    distance = np.min(
        [np.linalg.norm(point - v) for v in truth_proj[hull.vertices]]
    )  # Rough edge distance
    status = (
        "Deep in Truth Realm"
        if inside
        else (
            "Outside (Lie/Paradox)"
            if distance > 0.5
            else "On the Boundary (Conjecture Candidate!)"
        )
    )
    print(f"{tests[i][:60]}... | Inside Hull: {inside} | Status: {status}")

plt.title(
    f"Veraciscope Truth Polygon - qwen3-embedding:8b\nBlue Fill = Convex Hull of Proven Theorems"
)
plt.legend()
plt.savefig("./results/umap_truth_polygon_qwen3.png", dpi=200, bbox_inches="tight")
plt.show()

reducer = umap.UMAP(n_components=3, n_neighbors=15, min_dist=0.1, random_state=42)
proj_3d = reducer.fit_transform(all_emb)
truth_proj_3d = proj_3d[: len(truths)]

import alphashape
from matplotlib.path import Path

# Concave hull with tunable alpha (lower = tighter/more concave)
alpha = 1.5  # Tune this! Start 1-5; lower for craggier facets capturing conjectures
alpha_shape = alphashape.alphashape(truth_proj, alpha)

# Plot the concave polygon (filled + outline)
if hasattr(alpha_shape, "exterior"):
    x, y = alpha_shape.exterior.xy
    patch = plt.Polygon(
        list(zip(x, y)),
        closed=True,
        fill=True,
        alpha=0.3,
        color="blue",
        label="Truth Facets (Concave Hull)",
    )
    plt.gca().add_patch(patch)
    plt.plot(x, y, "b-", lw=2, alpha=0.8)

# Path for point-in-polygon checks
concave_path = Path(list(zip(x, y)))

# Classify tests with better status
for i, point in enumerate(test_proj):
    inside = concave_path.contains_point(point)
    # Rough distance to boundary
    distances = [np.linalg.norm(point - np.array([px, py])) for px, py in zip(x, y)]
    min_dist = min(distances) if distances else 999
    if inside:
        status = "Deep in Truth Realm (Proven Fit!)"
    elif min_dist < 0.5:  # Tune threshold based on plot scale
        status = "On the Facet Edge (Conjecture Candidate—Proof Needed!)"
    else:
        status = "Cast Into Shadow (Lie/Paradox/Rejected)"
    print(
        f"{tests[i][:70]}... | Inside: {inside} | Dist to Edge: {min_dist:.2f} | Status: {status}"
    )

plt.title(
    f"Veraciscope Truth Facets - qwen3-embedding:8b\nBlue = Concave Hull (alpha={alpha})"
)
plt.legend()
plt.savefig(
    "./results/umap_concave_truth_facets_qwen3.png", dpi=200, bbox_inches="tight"
)
plt.show()

import plotly.graph_objects as go

fig = go.Figure()

# Truth points
fig.add_trace(
    go.Scatter3d(
        x=truth_proj_3d[:, 0],
        y=truth_proj_3d[:, 1],
        z=truth_proj_3d[:, 2],
        mode="markers",
        marker=dict(size=3, color="blue", opacity=0.6),
        name="Proven Theorems",
    )
)

# Tests
test_proj_3d = proj_3d[-len(tests) :]
fig.add_trace(
    go.Scatter3d(
        x=test_proj_3d[:, 0],
        y=test_proj_3d[:, 1],
        z=test_proj_3d[:, 2],
        mode="markers+text",
        marker=dict(size=5, color="green"),
        text=[t[:20] for t in tests],
        name="Tests",
    )
)

# Optional: 3D convex hull mesh (fancy!)
from scipy.spatial import ConvexHull

hull_3d = ConvexHull(truth_proj_3d)
fig.add_trace(
    go.Mesh3d(
        x=truth_proj_3d[:, 0],
        y=truth_proj_3d[:, 1],
        z=truth_proj_3d[:, 2],
        i=hull_3d.simplices[:, 0],
        j=hull_3d.simplices[:, 1],
        k=hull_3d.simplices[:, 2],
        color="lightblue",
        opacity=0.2,
        name="3D Truth Polyhedron",
    )
)

fig.update_layout(
    title="3D Rotatable Veraciscope Truth Polyhedron", scene=dict(aspectmode="data")
)
fig.show()  # Opens interactive in browser—rotate/zoom!
fig.write_html("./results/3d_truth_polyhedron_qwen3.html")  # Shareable!
