from __future__ import annotations

import json
from typing import Optional, Union

from loguru import logger
from pydantic import BaseModel
from rich import print
from rich.console import Console
from rich.theme import Theme

custom_theme = Theme({"info": "dim cyan", "warning": "magenta", "danger": "bold red"})
console = Console(theme=custom_theme)


class AzureOpenAIConfig(BaseModel):
    """Read + Write configs for Azure OpenAI API to JSON file to put in .env"""

    api_version: str
    azure_endpoint: str
    azure_deployment: str
    api_key: str
    timeout: Optional[int] = None

    def to_json(self):
        """Developer helper tool to create a JSON file for manually adding to your local .env"""
        file_path = self.azure_deployment + "azure.json"
        with open(file_path, "w") as f:
            f.write(json.dumps(self.model_dump_json()))
        logger.success(f"Saved Azure OpenAI config to {file_path}")


class AzureConfig(BaseModel):
    """Simplified Azure config with shared account fields and per-role deployment names."""

    api_key: str
    api_version: str
    azure_endpoint: str
    embedding_deployment: str = "text-embedding-3-large"
    llm_deployment: str = "o3-mini"
    embedding_timeout: Optional[int] = 120
    llm_timeout: Optional[int] = None

    def _to_embedding_config(self) -> AzureOpenAIConfig:
        return AzureOpenAIConfig(
            api_key=self.api_key,
            api_version=self.api_version,
            azure_endpoint=self.azure_endpoint,
            azure_deployment=self.embedding_deployment,
            timeout=self.embedding_timeout,
        )

    def _to_llm_config(self) -> AzureOpenAIConfig:
        return AzureOpenAIConfig(
            api_key=self.api_key,
            api_version=self.api_version,
            azure_endpoint=self.azure_endpoint,
            azure_deployment=self.llm_deployment,
            timeout=self.llm_timeout,
        )


class LabeledDoc(BaseModel):
    pos: Optional[int] = None
    doc: str
    label: int
    prob: Optional[float] = None
    llm_label: Optional[str] = None


class Clusters(BaseModel):
    clusters: dict[Union[int, str], list[LabeledDoc]]
    bertopic_kwargs: dict
    embedding_llm_name: str

    def save(self, file_path: str):
        with open(file_path, "w") as f:
            f.write(json.dumps(self.model_dump_json(indent=4)))
        logger.success(f"Saved clusters to {file_path}")

    @classmethod
    def load(cls, file_path: str) -> Clusters:
        with open(file_path, "r") as f:
            data = json.load(f)
        return cls(**data)
