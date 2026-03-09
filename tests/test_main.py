import json
import os

import pytest
from nobs_clusters import nobs_cluster, nobs_cluster_azure
from nobs_clusters.classify_outliers import classify_outliers
from nobs_clusters.cluster import cluster
from nobs_clusters.embedding import embed
from nobs_clusters.input_examples import diet_actions
from nobs_clusters.main import nobs_cluster_azure
from nobs_clusters.models import AzureConfig, AzureOpenAIConfig
from nobs_clusters.naming import name
from dotenv import load_dotenv
from openai import AsyncAzureOpenAI, AsyncOpenAI, AzureOpenAI, OpenAI
from rich import print

load_dotenv()
openai = OpenAI(api_key=os.environ.get("OPENAI_API_KEY"))
async_openai = AsyncOpenAI(api_key=os.environ.get("OPENAI_API_KEY"))

## AZURE OPENAI CONFIG ##
from nobs_clusters.models import AzureOpenAIConfig
from openai import AzureOpenAI

# Below needs to be refactored in the future so we can remove hardcoded values
azure_openai_json = os.environ.get("text-embedding-3-large")
if azure_openai_json is None:
    raise ValueError(
        "add the AzureOpenAI's `text-embedding-3-large` config to .env file"
    )
azure_openai_config = AzureOpenAIConfig(**json.loads(azure_openai_json))

azure_embedding_client = AzureOpenAI(
    api_version=azure_openai_config.api_version,
    azure_endpoint=azure_openai_config.azure_endpoint,
    azure_deployment=azure_openai_config.azure_deployment,  # model name
    api_key=azure_openai_config.api_key,
    timeout=azure_openai_config.timeout,
)
azure_async_classifier_client = AsyncAzureOpenAI(
    api_version="2024-12-01-preview",
    azure_endpoint="https://boris-m3ndov9n-eastus2.openai.azure.com/",
    azure_deployment="o3-mini",  # model name
    api_key=azure_openai_config.api_key,
)
azure_naming_client = AzureOpenAI(
    api_version="2024-12-01-preview",
    azure_endpoint="https://boris-m3ndov9n-eastus2.openai.azure.com/",
    azure_deployment="o3-mini",  # model name
    api_key=azure_openai_config.api_key,
)


def test_embeddings():
    embedding_client = openai
    llm_model_name = "text-embedding-3-large"

    sample_size = 20
    embeddings = embed(
        texts=diet_actions[:sample_size],
        openai=embedding_client,
        llm_model_name=llm_model_name,
        with_disk_cache=False,
    )
    print(embeddings)


def test_embeddings_azure():
    llm_model_name = "text-embedding-3-large"

    sample_size = 20
    embeddings = embed(
        texts=diet_actions[:sample_size],
        openai=azure_embedding_client,
        llm_model_name=llm_model_name,
        with_disk_cache=False,
    )
    print(embeddings)


def test_embeddings_w_cache():
    embedding_client = openai
    llm_model_name = "text-embedding-3-large"
    sample_size = 20
    embeddings = embed(
        texts=diet_actions[:sample_size],
        openai=embedding_client,
        llm_model_name=llm_model_name,
        with_disk_cache=True,
    )
    print(embeddings)


@pytest.fixture(scope="session")
def test_clusters():
    llm_model_name = "text-embedding-3-large"
    clusters = cluster(
        bertopic_kwargs=dict(min_topic_size=4),
        docs=diet_actions,
        openai=openai,
        embed_llm_name=llm_model_name,
        with_disk_cache=True,
    )
    return clusters


@pytest.fixture(scope="session")
def test_clusters_azure_fixture():
    llm_model_name = "text-embedding-3-large"
    clusters = cluster(
        bertopic_kwargs=dict(min_topic_size=4),
        docs=diet_actions,
        openai=azure_embedding_client,
        embed_llm_name=llm_model_name,
        with_disk_cache=True,
    )
    return clusters


@pytest.fixture(scope="session")
def test_naming_fixture(test_clusters):
    named_clusters = name(
        clusters=test_clusters,
        openai=openai,
        llm_model_name="o3-mini",
        reasoning_effort="low",
        subject="personal diet intervention outcomes",
    )
    return named_clusters


def test_naming(test_clusters):
    named_clusters = name(
        clusters=test_clusters,
        openai=openai,
        llm_model_name="o3-mini",
        reasoning_effort="low",
        subject="personal diet intervention outcomes",
    )
    return named_clusters


@pytest.fixture(scope="session")
def test_naming_azure_fixture(test_clusters):
    named_clusters = name(
        clusters=test_clusters,
        openai=azure_naming_client,
        llm_model_name="o3-mini",
        reasoning_effort="low",
        subject="personal diet intervention outcomes",
    )
    return named_clusters


def test_naming_azure(test_clusters):
    named_clusters = name(
        clusters=test_clusters,
        openai=azure_naming_client,
        llm_model_name="o3-mini",
        reasoning_effort="low",
        subject="personal diet intervention outcomes",
    )
    return named_clusters


def test_classify_outliers(test_naming_fixture, test_clusters):
    merged = classify_outliers(
        named_clusters=test_naming_fixture,
        outliers=test_clusters.clusters[-1],
        openai=async_openai,
        llm_name="o3-mini",  # ONLY ONE ALLOWED
        reasoning_effort="low",
    )
    print(merged)


def test_classify_outliers_azure(
    test_naming_azure_fixture, test_clusters_azure_fixture
):
    merged = classify_outliers(
        named_clusters=test_naming_azure_fixture,
        outliers=test_clusters_azure_fixture.clusters[-1],
        openai=azure_async_classifier_client,
        llm_name="o3-mini",  # ONLY ONE ALLOWED
        reasoning_effort="low",
    )
    print(merged)


def test_nobs_cluster():
    openai_api_key = os.environ["OPENAI_API_KEY"]
    clusters = nobs_cluster(
        texts=diet_actions,
        openai_api_key=openai_api_key,
        reasoning_effort="low",
        subject="personal diet intervention outcomes",
    )
    print(clusters)


def test_nobs_cluster_azure():
    # Create Azure OpenAI configs for each component (legacy 3-config style)
    embedding_config = AzureOpenAIConfig(
        api_version=azure_openai_config.api_version,
        azure_endpoint=azure_openai_config.azure_endpoint,
        azure_deployment="text-embedding-3-large",
        api_key=azure_openai_config.api_key,
        timeout=azure_openai_config.timeout,
    )

    naming_config = AzureOpenAIConfig(
        api_version="2024-12-01-preview",
        azure_endpoint="https://boris-m3ndov9n-eastus2.openai.azure.com/",
        azure_deployment="o3-mini",
        api_key=azure_openai_config.api_key,
        timeout=60,
    )

    classifier_config = AzureOpenAIConfig(
        api_version="2024-12-01-preview",
        azure_endpoint="https://boris-m3ndov9n-eastus2.openai.azure.com/",
        azure_deployment="o3-mini",
        api_key=azure_openai_config.api_key,
        timeout=60,
    )

    # Run the nobs_cluster_azure function
    clusters = nobs_cluster_azure(
        texts=diet_actions,
        reasoning_effort="low",
        subject="personal diet intervention outcomes",
        azure_embeder_config=embedding_config,
        azure_namer_config=naming_config,
        azure_classifier_config=classifier_config,
    )

    print(clusters)


def test_nobs_cluster_azure_simple():
    """Test using the new simplified AzureConfig (single object instead of 3)."""
    config = AzureConfig(
        api_key=azure_openai_config.api_key,
        api_version=azure_openai_config.api_version,
        azure_endpoint=azure_openai_config.azure_endpoint,
        embedding_deployment="text-embedding-3-large",
        llm_deployment="o3-mini",
        embedding_timeout=120,
    )

    clusters = nobs_cluster_azure(
        texts=diet_actions,
        reasoning_effort="low",
        subject="personal diet intervention outcomes",
        azure_config=config,
    )

    print(clusters)
