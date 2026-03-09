import os
from typing import Literal, Optional

from nobs_clusters.classify_outliers import classify_outliers
from nobs_clusters.cluster import cluster
from nobs_clusters.input_examples import diet_actions
from nobs_clusters.models import AzureConfig, AzureOpenAIConfig, Clusters
from nobs_clusters.naming import name
from dotenv import load_dotenv
from loguru import logger
from openai import AsyncAzureOpenAI, AsyncOpenAI, AzureOpenAI, OpenAI
from rich import print

load_dotenv()


def nobs_cluster(
    *,
    texts: list[str],
    openai_api_key: str,
    reasoning_effort: Literal["low", "medium", "high"],
    subject: str,
    embed_model: str = "text-embedding-3-large",
    llm_model: str = "o3-mini",
) -> Clusters:
    if len(texts) < 4:
        raise ValueError("Need at least 4 texts to cluster")
    openai = OpenAI(api_key=openai_api_key)
    async_openai = AsyncOpenAI(api_key=openai_api_key)
    clusters = cluster(
        bertopic_kwargs=dict(min_topic_size=4),
        docs=texts,
        openai=openai,
        embed_llm_name=embed_model,
        with_disk_cache=True,
    )
    named_clusters = name(
        clusters=clusters,
        openai=openai,
        llm_model_name=llm_model,
        reasoning_effort=reasoning_effort,
        subject=subject,
    )
    try:
        merged = classify_outliers(
            named_clusters=named_clusters,
            outliers=clusters.clusters[-1],
            openai=async_openai,
            llm_name=llm_model,
            reasoning_effort=reasoning_effort,
        )
    except KeyError:
        logger.debug("No outliers found")
        return named_clusters
    return merged


def nobs_cluster_azure(
    *,
    texts: list[str],
    reasoning_effort: Literal["low", "medium", "high"],
    subject: str,
    azure_config: Optional[AzureConfig] = None,
    azure_embedder_config: Optional[AzureOpenAIConfig] = None,
    azure_embeder_config: Optional[AzureOpenAIConfig] = None,
    azure_namer_config: Optional[AzureOpenAIConfig] = None,
    azure_classifier_config: Optional[AzureOpenAIConfig] = None,
) -> Clusters:
    if len(texts) < 4:
        raise ValueError("Need at least 4 texts to cluster")

    if azure_config is not None:
        embed_cfg = azure_config._to_embedding_config()
        llm_cfg = azure_config._to_llm_config()
    else:
        embed_cfg = azure_embedder_config or azure_embeder_config
        llm_cfg = azure_namer_config
        if embed_cfg is None or llm_cfg is None:
            raise ValueError(
                "Provide either azure_config or all three legacy config params"
            )
        if azure_classifier_config is None:
            azure_classifier_config = llm_cfg

    classifier_cfg = azure_classifier_config if azure_config is None else llm_cfg

    clusters = cluster(
        bertopic_kwargs=dict(min_topic_size=4),
        docs=texts,
        openai=AzureOpenAI(**embed_cfg.model_dump()),
        embed_llm_name=embed_cfg.azure_deployment,
        with_disk_cache=True,
    )
    named_clusters = name(
        clusters=clusters,
        openai=AzureOpenAI(**llm_cfg.model_dump()),
        llm_model_name=llm_cfg.azure_deployment,
        reasoning_effort=reasoning_effort,
        subject=subject,
    )
    try:
        merged = classify_outliers(
            named_clusters=named_clusters,
            outliers=clusters.clusters[-1],
            openai=AsyncAzureOpenAI(**classifier_cfg.model_dump()),
            llm_name=classifier_cfg.azure_deployment,
            reasoning_effort=reasoning_effort,
        )
    except KeyError:
        logger.debug("No outliers found")
        return named_clusters
    return merged
