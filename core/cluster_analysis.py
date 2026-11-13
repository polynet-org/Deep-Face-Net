"""
core/cluster_analysis.py
========================
K-Means based face-embedding clustering, ported from Deep-Live-Cam 2.7.

Used by the Map Faces pipeline to automatically discover the unique identities
present across the frames of a target video and assign each frame-face to the
closest cluster centroid (identity).
"""

from typing import Any, List, Tuple, Optional

import numpy as np
from sklearn.cluster import KMeans


# ---------------------------------------------------------------------------
# Public helpers
# ---------------------------------------------------------------------------

def find_cluster_centroids(
    embeddings: List[np.ndarray],
    max_k: int = 10,
) -> np.ndarray:
    """Use K-Means with elbow-method selection to find the optimal number of
    unique face identities in *embeddings*.

    Parameters
    ----------
    embeddings:
        List of L2-normalised face embeddings (one per detected face, across
        all sampled frames of the target video).
    max_k:
        Upper bound on the number of clusters to test.

    Returns
    -------
    np.ndarray  shape (k_optimal, embedding_dim)
        Cluster centroids, one row per unique identity discovered.
    """
    if not embeddings:
        return np.array([])

    n = len(embeddings)
    max_k = min(max_k, n)          # can't have more clusters than samples

    if max_k == 1:
        # Only one face — trivial, no need to cluster
        return np.mean(embeddings, axis=0, keepdims=True)

    inertia: List[float] = []
    cluster_centroids: List[dict] = []

    for k in range(1, max_k + 1):
        kmeans = KMeans(n_clusters=k, random_state=0, n_init="auto")
        kmeans.fit(embeddings)
        inertia.append(float(kmeans.inertia_))
        cluster_centroids.append({"k": k, "centroids": kmeans.cluster_centers_})

    # Elbow method: pick k where the inertia reduction is largest
    diffs = [inertia[i] - inertia[i + 1] for i in range(len(inertia) - 1)]
    best_idx = diffs.index(max(diffs)) + 1        # +1 because diffs is offset
    return cluster_centroids[best_idx]["centroids"]


def find_closest_centroid(
    centroids: Any,
    normed_face_embedding: np.ndarray,
) -> Tuple[int, np.ndarray]:
    """Return the index and vector of the centroid most similar to the given
    face embedding (cosine similarity via dot product on normalised vectors).

    Parameters
    ----------
    centroids:
        Array-like (n_centroids, embedding_dim) of cluster centres.
    normed_face_embedding:
        L2-normalised embedding of the face to match.

    Returns
    -------
    (index, centroid_vector)
    """
    centroids_arr = np.array(centroids)
    emb = np.array(normed_face_embedding)
    similarities = np.dot(centroids_arr, emb)
    idx = int(np.argmax(similarities))
    return idx, centroids_arr[idx]


def build_source_target_map(
    source_target_map: List[dict],
) -> dict:
    """Flatten a full source_target_map into the simplified form used during
    live face-matching: { 'source_faces': [...], 'target_embeddings': [...] }.

    Only entries that have *both* a 'source' and a 'target' key are included.
    """
    source_faces = []
    target_embeddings = []

    for entry in source_target_map:
        if "source" in entry and "target" in entry:
            source_face = entry["source"].get("face")
            target_face = entry["target"].get("face")
            if source_face is not None and target_face is not None:
                source_faces.append(source_face)
                target_embeddings.append(target_face.normed_embedding)

    return {"source_faces": source_faces, "target_embeddings": target_embeddings}
