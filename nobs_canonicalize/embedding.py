"""
Source:
    https://github.com/UKPLab/sentence-transformers/blob/master/examples/applications/clustering/fast_clustering.py
"""

import time
from dataclasses import dataclass
from operator import attrgetter
from typing import Optional, Union

import more_itertools
import numpy as np
from diskcache import Cache
from loguru import logger
from openai import AzureOpenAI, OpenAI
from rich import print

cache_path = "./cache_embeddings"
cache = Cache(cache_path)


def clear_texts(texts: list[str]):
    """Opinionated text cleaning"""
    clean_sentences = []
    for text in texts:
        if text == "":
            clean_sentences.append("NONE")
            # breakpoint()
        elif text is not None and len(text) > 0:
            clean_sentences.append(text)
        else:
            clean_sentences.append("NONE")
    clean_sentences = [text.strip() for text in clean_sentences]
    clean_sentences = [text.replace("\n", "") for text in clean_sentences]
    return clean_sentences


def _embed(
    *,
    texts: list[str],
    openai: Union[OpenAI, AzureOpenAI],
    llm_model_name: str,
):
    clean_sentences = clear_texts(texts)

    big_list_of_embeddings = []
    batch_size = 100
    text_batches = list(more_itertools.batched(clean_sentences, batch_size))
    for idx, task_batch in enumerate(text_batches):
        try:
            response = openai.embeddings.create(input=task_batch, model=llm_model_name)
            pause_time = 1
            logger.info(f"On {idx} of {batch_size} - sleeping for {pause_time} seconds")
            time.sleep(pause_time)
        except Exception as e:
            # TODO - handle this better
            raise e

        list_of_embeddings = [d.embedding for d in response.data]
        big_list_of_embeddings.extend(list_of_embeddings)
    embeddings_array = np.array(big_list_of_embeddings)
    logger.info(f"Created embeddings numpy array, with shape: {embeddings_array.shape}")
    return embeddings_array


def _embed_w_cache(
    *, texts: list[str], openai: Union[OpenAI, AzureOpenAI], llm_model_name: str
):
    """Cached the embeddings of the texts in disk"""
    logger.warning("Using disk cache for embeddings")

    @dataclass
    class Result:
        text: str
        position: int
        embedding: Optional[list[float]] = None

    complete_results = []
    incomplete_results = []
    for position, text in enumerate(texts):
        embedded_text = cache.get(text)
        if embedded_text is not None:
            # logger.info(f"Cache hit for text: {text}")
            complete_results.append(
                Result(text=text, position=position, embedding=embedded_text)
            )
        else:
            # logger.debug(f"Cache miss for text: {text}")
            incomplete_results.append(Result(text=text, position=position))
    logger.info(f"Cache hit for {len(complete_results)} texts")
    logger.info(f"Cache miss for {len(incomplete_results)} texts")
    if len(incomplete_results) == 0:
        just_embeddings = [r.embedding for r in complete_results]
        return np.array(just_embeddings)
    else:
        logger.info(f"Requesting embeddings for {len(incomplete_results)} texts")
        texts = [r.text for r in incomplete_results]
        cloud_results = embed(
            texts=texts,
            openai=openai,
            llm_model_name=llm_model_name,
            with_disk_cache=False,
        )
        for incomplete_result, embedding in zip(incomplete_results, cloud_results):
            cache[incomplete_result.text] = embedding

        for incomplete_result, embedding in zip(incomplete_results, cloud_results):
            incomplete_result.embedding = embedding
            complete_results.append(incomplete_result)
        # sort the completes results by position
        complete_results.sort(key=attrgetter("position"))
        just_embeddings = [r.embedding for r in complete_results]
        logger.info(
            f"Returning {len(just_embeddings)} embeddings, cached in disk {cache_path}"
        )
        return np.array(just_embeddings)


def embed(
    *,
    texts: list[str],
    openai: Union[OpenAI, AzureOpenAI],
    llm_model_name: str,
    with_disk_cache: bool,
):
    """Embeds the texts using the client"""
    if with_disk_cache:
        return _embed_w_cache(texts=texts, openai=openai, llm_model_name=llm_model_name)
    else:
        return _embed(texts=texts, openai=openai, llm_model_name=llm_model_name)
