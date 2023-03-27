"""Generate embeddings with sentence transformers or through openai api"""
import logging
import os
from typing import Dict, List, Optional, Tuple

import openai
import pandas as pd
import torch
from dotenv import load_dotenv
from openai.embeddings_utils import get_embedding
from sentence_transformers import SentenceTransformer, util
from transformers import GPT2TokenizerFast

from data_types import ModelEnum

load_dotenv()
OPENAI_API_KEY = os.getenv("OPENAI_API_KEY")

openai.api_key = OPENAI_API_KEY


class EmbeddingGenerator:
    """Generate embeddings."""

    def __init__(self, model_name: ModelEnum) -> None:

        self.model = model_name
        # embedding model parameters
        self.embedding_model = "text-embedding-ada-002"
        self.embedding_encoding = (
            "cl100k_base"  # this the encoding for text-embedding-ada-002
        )
        self.max_tokens = 8000  # the maximum for text-embedding-ada-002 is 8191
        self.embeddings: Optional[torch.Tensor] = None
        self.query_embeddings: Optional[torch.Tensor] = None

    def embed_data(self, data: pd.DataFrame) -> None:
        """Embed data.
        Args:
            data (pd.Series): data to embed
        Returns:
            pd.Series: embedded data.
        """
        if self.model == ModelEnum.MINILM:
            logging.info("Embedding data with local model")
            embedder = SentenceTransformer("all-MiniLM-L6-v2")

            data_embeddings = embedder.encode(data["resumes"], convert_to_tensor=True)
            self.embeddings = data_embeddings
        elif self.model == ModelEnum.ADA2:
            logging.warning("Embedding data openai api; fees may apply!")
            data_embeddings = data["resumes"].apply(
                lambda x: get_embedding(x, engine=self.embedding_model)
            )
            self.embeddings = torch.Tensor(data_embeddings)
        else:
            raise ValueError("The embedding model has not been set!")

    def embeddings_exist(self) -> bool:
        """Check if embeddings exist for data.
        Returns:
            bool: True if embeddings exist, else False
        """
        if self.embeddings is None:
            return False
        return True

    def embed_queries(self, data=pd.DataFrame) -> None:
        """Embed query strings.
        Raises:
            ValueError: Raises ValueError is model has not been set.
        """
        if self.model == ModelEnum.MINILM:
            logging.info("Embedding query data with local model")
            embedder = SentenceTransformer("all-MiniLM-L6-v2")

            data_embeddings = embedder.encode(data["query"], convert_to_tensor=True)
            self.query_embeddings = data_embeddings
        elif self.model == ModelEnum.ADA2:
            logging.warning("Embedding query data openai api; fees may apply!")
            data_embeddings = data["query"].apply(
                lambda x: get_embedding(x, engine=self.embedding_model)
            )
            self.query_embeddings = torch.Tensor(data_embeddings)
        else:
            raise ValueError("The embedding model has not been set!")

    def save_embeddings(self) -> None:
        """Save embeddings to .pt file."""
        torch.save(self.embeddings, f"data/{self.model.value}-embeddings.pt")

    def load_embeddings(self, data_to_embed: pd.DataFrame) -> None:
        """Load embeddings from .pt file"""
        logging.info("Loading embeddings")
        try:
            self.embeddings = torch.load(f"data/{self.model.value}-embeddings.pt")
        except FileNotFoundError as exc:
            logging.warning(exc)
            logging.warning("Precomputed embeddings don't exist!")
        if not self.embeddings_exist():
            self.embed_data(data_to_embed)
            self.save_embeddings()

    def save_query_embeddings(self) -> None:
        """Save embeddings to .pt file."""
        torch.save(
            self.query_embeddings, f"data/{self.model.value}-query-embeddings.pt"
        )

    def load_query_embeddings(self) -> None:
        """Load embeddings from .pt file"""
        self.query_embeddings = torch.load(
            f"data/{self.model.value}-query-embeddings.pt"
        )

    def estimate_tokens(self, data: pd.DataFrame) -> int:
        """Estimate the number of tokens in the data.
        Args:
            data (pd.DataFrame): data for which to estimate number of tokens.
        Returns:
            int: number of tokens as per given by GPT2
        """
        tokenizer = GPT2TokenizerFast.from_pretrained("gpt2")
        tokens = 0
        for sentence in data["resumes"]:
            tokens += num_tokens_from_string(sentence, tokenizer)
        return tokens

    def estimate_price(self, tokens: int) -> float:
        """Estimate the price of an api call to openai embedding model.
        Args:
            num_of_tokens (int): num of tokens to send to openai api.
        Returns:
            float: estimated price of the api call.
        """
        prize_per_api_call = 0.0004  # per 1k tokens
        return prize_per_api_call * tokens / 1000

    def semantic_search(self, top_k: int) -> List[Dict]:
        """Perform semantic search on embeddings.
        Args:
            top_k (int): how many hits to return
        Returns:
            List[Dict]: results of the search
        """
        hits = util.semantic_search(self.query_embeddings, self.embeddings, top_k=top_k)
        return hits


def num_tokens_from_string(string: str, tokenizer) -> int:
    """Estimate number of tokens with given tokenizer.
    Args:
        string (str): string for which tokens are estimated
        tokenizer (_type_): tokenizer to use
    Returns:
        int: tokens in the string
    """
    return len(tokenizer.encode(string))


def read_csv_data() -> pd.DataFrame:
    """Read csv data in pd.DataFrame objects.
    Returns:
        tuple: data in dataframes
    """
    data_df = pd.read_parquet("data/df.resumes.gzip")
    data_df = data_df.drop_duplicates(subset="resumes")
    data_df = data_df.reset_index()

    return data_df


if __name__ == "__main__":

    df = read_csv_data()

    embedding_generator = EmbeddingGenerator(model_name=ModelEnum.MINILM)
    num_of_tokens = embedding_generator.estimate_tokens(df)
    estimated_price = embedding_generator.estimate_price(num_of_tokens)
    print(estimated_price)

    # embedding_generator.embed_data(data=df)
    # embedding_generator.save_embeddings()
    embedding_generator.load_embeddings(df)

    # embedding_generator.embed_queries(data=query_df)
    # embedding_generator.save_query_embeddings()
    embedding_generator.load_query_embeddings()

    print(embedding_generator.semantic_search(top_k=5))
