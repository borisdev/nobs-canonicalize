from copy import deepcopy
from typing import Literal, Optional, Union

import instructor
from nobs_canonicalize.models import Clusters
from loguru import logger
from openai import AzureOpenAI, OpenAI
from pydantic import BaseModel
from rich import print


def name(
    *,
    clusters: Clusters,
    openai: Union[OpenAI, AzureOpenAI],
    llm_model_name: str,
    reasoning_effort: Literal["low", "medium", "high"],
    subject: str,  #  = "personal diet intervention outcomes",
    prompt: Optional[str] = None,  # todo help downstream evaluation
) -> Clusters:
    if prompt is not None:
        raise NotImplementedError("Prompt is not implemented yet")
    named_clusters = {}
    if len(clusters.clusters) == 0:
        raise ValueError("Clusters is empty")

    class UnamedGroup(BaseModel):
        group_number: int
        docs: list[str]

    class UnamedGroups(BaseModel):
        unnamed_groups: list[UnamedGroup]

    unamed_groups = []
    for group_number, members in clusters.clusters.items():
        if group_number == -1:  # skip outliers
            continue
        # create pretty input of what needs naming to embed in the prompt for the LLM to read
        unamed_group = UnamedGroup(
            group_number=group_number, docs=[member.doc for member in members]
        )
        unamed_groups.append(unamed_group)
    unamed_groups_json = UnamedGroups(unnamed_groups=unamed_groups).model_dump_json(
        indent=2
    )

    prompt = f"""

    Instructions
    ========================

    You are a university research assistant working on a project about {subject}.

    Provide a very informative research headline for each of the following unnamed groups of {subject}.

    The headlines must be distinct from one another.

    Get to the point quickly. Don't use unnecessary words. Don't use filler words. Don't use vague words.

    Don't use words that are not necessary to convey the meaning of the group such as overly general words
    or generic terms.

    Specifics in the headline are good so people can quickly understand what the group is about.

    Make sure the headline represents the common specifics of what makes its group different that the other groups.

    Input language context
    ----------------------

    Remember this is for scientific research on what ordinary people are
    discussing about on {subject} on Reddit, therefore, you must handle text
    input that relates to sexual content or self-harm in a professional manner
    and continue to provide a headline that is informative. Don't be offended.
    There is never a reason to jump to a conclusion about danger since all the
    data is anonymized and purely for research purposes.


    Input
    ========================

    {unamed_groups_json}

    """

    class NamedGroup(BaseModel):
        group_number: int
        group_headline: str

    class NamedGroups(BaseModel):
        named_groups: list[NamedGroup]

    llm_client = instructor.patch(openai)

    completion_config = {
        "model": llm_model_name,
        "messages": [{"role": "user", "content": prompt}],
        "response_model": NamedGroups,
        "reasoning_effort": reasoning_effort,
    }
    logger.info("Waiting for completion from LLM  -- might take a couple of minutes")
    response = llm_client.chat.completions.create(**completion_config)
    if response is Exception:
        logger.error(response)
        raise
    groups_number2group_headline = {
        named_group.group_number: named_group.group_headline
        for named_group in response.named_groups
    }
    named_clusters = {}
    for number, headline in groups_number2group_headline.items():
        named_clusters[headline] = [member for member in clusters.clusters[number]]
    results_clusters = deepcopy(clusters)
    results_clusters.clusters = named_clusters
    return results_clusters
