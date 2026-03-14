import asyncio
from copy import deepcopy
from typing import Literal, Optional, Union

from loguru import logger
from openai import AsyncAzureOpenAI, AsyncOpenAI
from pydantic import BaseModel, Field

from nobs_canonicalize.chain import Chain
from nobs_canonicalize.models import Clusters, LabeledDoc


async def _classify_outliers(
    openai: Union[AsyncOpenAI, AsyncAzureOpenAI],
    named_clusters: Clusters,
    outliers: list,
    start: int,
    size: Union[int, None],
    batch_size: int,
    llm_name: str,
    max_retries: int,
    reasoning_effort: Literal["high", "medium", "low"],
) -> Clusters:
    class Input(BaseModel):
        text: str

    labels = list(named_clusters.clusters.keys())
    class MultiClassPredictionResponse(BaseModel):

        tag: Optional[Literal[*labels]] = Field(
            default=None,
            title="Diet Tags",
            description="""
                Tag the text with the best diet tag based on your scientific knowledge
                of using changes in diet as an intervention to improve health outcomes.
                If you are not sure leave it blank.
            """,
        )
    class MultiClassChain(Chain):

        input_schema = Input
        output_schema = MultiClassPredictionResponse

        @classmethod
        def make_input_text(cls, *, input: Input) -> str:
            input_text = f"""

                {input.text}

            """
            return input_text
    if size is None:
        size = len(outliers) - start

    input_docs = outliers[start : start + size]
    input_objects = [Input(text=doc.doc) for doc in input_docs]
    logger.info(f"Classifying {len(input_objects)} outliers")

    responses = await MultiClassChain.batch_predict(
        openai=openai,
        size=batch_size,
        llm_name=llm_name,
        input_objects=input_objects,
        max_retries=max_retries,
        reasoning_effort=reasoning_effort,
    )
    merged = deepcopy(named_clusters)
    for response, input_object in zip(responses, input_objects):
        # 3 cases: 1) response is an exception, 2) response.tag is None (unclassified), 3) response.tag is a class label
        if isinstance(response, Exception):
            logger.warning(f"Error classifying outlier: {input_object.text} => {str(response)}")
            llm_label = f"o3-mini:{reasoning_effort}-error-{str(response)}"
            class_label = "unclassified"
        else:
            if response.tag is None:
                llm_label = f"o3-mini:{reasoning_effort}-unclassified"
                class_label = "unclassified"
                print(input_object.text, "=>", response.tag)
            else:
                llm_label = f"o3-mini:{reasoning_effort}"
                class_label = response.tag
                print(input_object.text, "=>", response.tag)
        labeled_doc = LabeledDoc(doc=input_object.text, label=-1, llm_label=llm_label)
        try:
            merged.clusters[class_label].append(labeled_doc)
        except KeyError:
            merged.clusters[class_label] = [labeled_doc]

    return merged

def classify_outliers(
    named_clusters: Clusters,
    outliers: list,
    openai: Union[AsyncOpenAI, AsyncAzureOpenAI],
    reasoning_effort: Literal["high", "medium", "low"],
    llm_name: str,
    start: int = 0,
    size: Union[int, None] = None,
    batch_size: int = 100,
    max_retries: int = 3,
) -> Clusters:
    if outliers is None or len(outliers) == 0:
        logger.warning("No outliers to classify")
        return named_clusters
    labels = list(named_clusters.clusters.keys())
    if len(labels) == 0:
        logger.warning("No topics (cluster names) to classify outliers against")
        return named_clusters
    return asyncio.run(
        _classify_outliers(  # type: ignore
            openai=openai,
            llm_name=llm_name,
            outliers=outliers,
            named_clusters=named_clusters,
            start=start,
            size=size,
            batch_size=batch_size,
            max_retries=max_retries,
            reasoning_effort=reasoning_effort,
        )
    )
