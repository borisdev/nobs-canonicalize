from __future__ import annotations

from collections import defaultdict
from typing import Union

import faiss
import igraph as ig
import numpy as np
from loguru import logger
from openai import AzureOpenAI, OpenAI
from rich import print

from nobs_canonicalize.embedding import embed
from nobs_canonicalize.models import Clusters, LabeledDoc


def cluster_faiss_leiden(
    *,
    docs: list[str],
    openai: Union[AzureOpenAI, OpenAI],
    embed_llm_name: str,
    with_disk_cache: bool,
    n_neighbors: int = 20,
    min_sim: float = 0.55,
    resolution: float = 0.1,
    min_cluster_size: int = 4,
) -> Clusters:
    """Cluster documents using FAISS kNN graph + Leiden community detection.

    Args:
        n_neighbors: Number of nearest neighbors for kNN graph.
        min_sim: Minimum cosine similarity to keep an edge.
        resolution: Leiden resolution parameter (CPM objective).
        min_cluster_size: Clusters smaller than this become outliers (-1).
    """
    if len(docs) < 4:
        print(docs)
        raise ValueError("Need at least 4 texts to cluster")

    embeddings = embed(
        texts=docs,
        openai=openai,
        llm_model_name=embed_llm_name,
        with_disk_cache=with_disk_cache,
    )

    # Normalize for cosine similarity via inner product
    X = np.asarray(embeddings, dtype="float32").copy()
    faiss.normalize_L2(X)

    n, d = X.shape
    k = min(n_neighbors, n - 1)  # can't have more neighbors than docs

    # Flat inner product index (cosine on normalized vectors)
    # Use HNSW for large datasets (>50K); flat is exact and reliable for smaller ones
    if n > 50_000:
        M = 32
        index = faiss.IndexHNSWFlat(d, M, faiss.METRIC_INNER_PRODUCT)
    else:
        index = faiss.IndexFlatIP(d)
    index.add(X)

    # Search k+1 because the query point itself appears as a neighbor
    scores, nbrs = index.search(X, k + 1)

    # Build sparse graph with similarity threshold
    edges = []
    weights = []
    for i in range(n):
        for score, j in zip(scores[i], nbrs[i]):
            if j < 0 or j == i:
                continue
            if score >= min_sim:
                edges.append((i, int(j)))
                weights.append(float(score))

    if not edges:
        logger.warning("No edges above min_sim threshold; all docs are outliers")
        labeled_docs = [LabeledDoc(doc=doc, label=-1) for doc in docs]
        return Clusters(
            clusters={-1: labeled_docs},
            bertopic_kwargs={
                "backend": "faiss_leiden",
                "n_neighbors": n_neighbors,
                "min_sim": min_sim,
                "resolution": resolution,
                "min_cluster_size": min_cluster_size,
            },
            embedding_llm_name=embed_llm_name,
        )

    g = ig.Graph(n=n, edges=edges, directed=False)
    g.es["weight"] = weights
    g.simplify(combine_edges="max")

    partition = g.community_leiden(
        weights="weight",
        resolution=resolution,
        objective_function="CPM",
    )

    # Assign labels; small clusters become outliers
    labels = list(partition.membership)
    cluster_sizes = defaultdict(int)
    for label in labels:
        cluster_sizes[label] += 1

    # Remap: small clusters → -1, renumber the rest contiguously
    valid_clusters = {
        old_id for old_id, size in cluster_sizes.items() if size >= min_cluster_size
    }
    remap = {}
    next_id = 0
    for old_id in sorted(valid_clusters):
        remap[old_id] = next_id
        next_id += 1

    final_labels = []
    for label in labels:
        if label in remap:
            final_labels.append(remap[label])
        else:
            final_labels.append(-1)

    labeled_docs = [
        LabeledDoc(doc=doc, label=label)
        for doc, label in zip(docs, final_labels)
    ]

    clusters = defaultdict(list)
    for ld in labeled_docs:
        clusters[int(ld.label)].append(ld)

    result = Clusters(
        clusters=clusters,
        bertopic_kwargs={
            "backend": "faiss_leiden",
            "n_neighbors": n_neighbors,
            "min_sim": min_sim,
            "resolution": resolution,
            "min_cluster_size": min_cluster_size,
        },
        embedding_llm_name=embed_llm_name,
    )
    print(result)

    n_clusters = len([k for k in result.clusters if k != -1])
    n_outliers = len(result.clusters.get(-1, []))
    logger.info(f"FAISS+Leiden: {n_clusters} clusters, {n_outliers} outliers")

    if result.clusters is None or len(result.clusters) == 0:
        raise ValueError("No clusters found")
    if len(result.clusters) == 1 and -1 in result.clusters:
        logger.warning(
            f"All {len(docs)} documents were assigned to the outlier cluster"
        )
    return result
