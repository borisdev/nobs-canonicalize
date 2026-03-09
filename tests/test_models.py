"""Unit tests for models and config logic — no API keys required."""

import pytest
from bertopic_easy.models import AzureConfig, AzureOpenAIConfig


class TestAzureConfig:
    def test_defaults(self):
        cfg = AzureConfig(
            api_key="key",
            api_version="2024-12-01-preview",
            azure_endpoint="https://example.openai.azure.com/",
        )
        assert cfg.embedding_deployment == "text-embedding-3-large"
        assert cfg.llm_deployment == "o3-mini"
        assert cfg.embedding_timeout == 120
        assert cfg.llm_timeout is None

    def test_custom_deployments(self):
        cfg = AzureConfig(
            api_key="key",
            api_version="2024-12-01-preview",
            azure_endpoint="https://example.openai.azure.com/",
            embedding_deployment="custom-embed",
            llm_deployment="custom-llm",
            embedding_timeout=60,
            llm_timeout=30,
        )
        assert cfg.embedding_deployment == "custom-embed"
        assert cfg.llm_deployment == "custom-llm"
        assert cfg.embedding_timeout == 60
        assert cfg.llm_timeout == 30

    def test_to_embedding_config(self):
        cfg = AzureConfig(
            api_key="test-key",
            api_version="2024-12-01-preview",
            azure_endpoint="https://example.openai.azure.com/",
            embedding_deployment="text-embedding-3-large",
            embedding_timeout=120,
        )
        embed_cfg = cfg._to_embedding_config()
        assert isinstance(embed_cfg, AzureOpenAIConfig)
        assert embed_cfg.api_key == "test-key"
        assert embed_cfg.api_version == "2024-12-01-preview"
        assert embed_cfg.azure_endpoint == "https://example.openai.azure.com/"
        assert embed_cfg.azure_deployment == "text-embedding-3-large"
        assert embed_cfg.timeout == 120

    def test_to_llm_config(self):
        cfg = AzureConfig(
            api_key="test-key",
            api_version="2024-12-01-preview",
            azure_endpoint="https://example.openai.azure.com/",
            llm_deployment="o3-mini",
            llm_timeout=30,
        )
        llm_cfg = cfg._to_llm_config()
        assert isinstance(llm_cfg, AzureOpenAIConfig)
        assert llm_cfg.api_key == "test-key"
        assert llm_cfg.api_version == "2024-12-01-preview"
        assert llm_cfg.azure_endpoint == "https://example.openai.azure.com/"
        assert llm_cfg.azure_deployment == "o3-mini"
        assert llm_cfg.timeout == 30

    def test_embedding_and_llm_configs_differ(self):
        """Ensure the two derived configs use different deployments/timeouts."""
        cfg = AzureConfig(
            api_key="key",
            api_version="v1",
            azure_endpoint="https://example.openai.azure.com/",
            embedding_deployment="embed-model",
            llm_deployment="llm-model",
            embedding_timeout=120,
            llm_timeout=60,
        )
        embed_cfg = cfg._to_embedding_config()
        llm_cfg = cfg._to_llm_config()
        assert embed_cfg.azure_deployment == "embed-model"
        assert llm_cfg.azure_deployment == "llm-model"
        assert embed_cfg.timeout == 120
        assert llm_cfg.timeout == 60
        # Shared fields must match
        assert embed_cfg.api_key == llm_cfg.api_key
        assert embed_cfg.api_version == llm_cfg.api_version
        assert embed_cfg.azure_endpoint == llm_cfg.azure_endpoint


class TestAzureOpenAIConfig:
    def test_creation(self):
        cfg = AzureOpenAIConfig(
            api_version="v1",
            azure_endpoint="https://example.openai.azure.com/",
            azure_deployment="deploy",
            api_key="key",
        )
        assert cfg.timeout is None

    def test_model_dump_for_client(self):
        """Ensure model_dump produces kwargs compatible with AzureOpenAI client."""
        cfg = AzureOpenAIConfig(
            api_version="v1",
            azure_endpoint="https://example.openai.azure.com/",
            azure_deployment="deploy",
            api_key="key",
            timeout=60,
        )
        d = cfg.model_dump()
        assert set(d.keys()) == {
            "api_version",
            "azure_endpoint",
            "azure_deployment",
            "api_key",
            "timeout",
        }


class TestImports:
    def test_top_level_imports(self):
        """Verify __init__.py exports the right symbols."""
        from bertopic_easy import (
            AzureConfig,
            AzureOpenAIConfig,
            Clusters,
            bertopic_easy,
            bertopic_easy_azure,
        )
        assert callable(bertopic_easy)
        assert callable(bertopic_easy_azure)

    def test_azure_config_importable_from_top(self):
        from bertopic_easy import AzureConfig
        cfg = AzureConfig(
            api_key="k",
            api_version="v",
            azure_endpoint="https://x.openai.azure.com/",
        )
        assert cfg.embedding_deployment == "text-embedding-3-large"


class TestBertopicEasyAzureValidation:
    def test_rejects_too_few_texts(self):
        from bertopic_easy import bertopic_easy_azure, AzureConfig

        cfg = AzureConfig(
            api_key="k",
            api_version="v",
            azure_endpoint="https://x.openai.azure.com/",
        )
        with pytest.raises(ValueError, match="at least 4 texts"):
            bertopic_easy_azure(
                texts=["a", "b"],
                reasoning_effort="low",
                subject="test",
                azure_config=cfg,
            )

    def test_rejects_missing_legacy_configs(self):
        from bertopic_easy import bertopic_easy_azure

        with pytest.raises(ValueError, match="azure_config"):
            bertopic_easy_azure(
                texts=["a", "b", "c", "d", "e"],
                reasoning_effort="low",
                subject="test",
                azure_embeder_config=AzureOpenAIConfig(
                    api_key="k",
                    api_version="v",
                    azure_endpoint="https://x.openai.azure.com/",
                    azure_deployment="d",
                ),
                # missing azure_namer_config
            )


class TestBertopicEasyValidation:
    def test_rejects_too_few_texts(self):
        from bertopic_easy import bertopic_easy

        with pytest.raises(ValueError, match="at least 4 texts"):
            bertopic_easy(
                texts=["a"],
                openai_api_key="fake-key",
                reasoning_effort="low",
                subject="test",
            )
