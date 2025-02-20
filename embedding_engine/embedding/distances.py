from abc import ABC, abstractmethod

import numpy as np
from sqlalchemy import select
from sqlalchemy.orm import Session

from embedding_engine.config import Config
from embedding_engine.database import PhrasesTable
from embedding_engine.embedding.compute import compute_phrase_embedding
from embedding_engine.embedding.tokenizer import Tokenizer


class PhrasesRetriever(ABC):
    """Retrieve nearest phrase to the embedding, calculate the corresponding distance."""

    def __init__(self, tokenizer: Tokenizer):
        self.tokenizer = tokenizer

    @abstractmethod
    def get_distance_column(self, embedding: np.ndarray):
        pass

    @abstractmethod
    def compute_distance(self, embedding1: np.ndarray, embedding2: np.ndarray) -> float:
        pass

    def get_nearest(self, embedding: np.ndarray, session: Session, limit: int = 1) -> None | tuple[str, float]:
        distance_column = self.get_distance_column(embedding)
        statement = (
            select(PhrasesTable.phrase, PhrasesTable.embedding, distance_column).order_by(distance_column).limit(limit)
        )
        result = session.execute(statement=statement)

        value = result.fetchone()
        if value is None:
            return None

        nearest, _, dist = value

        return nearest, dist

    def get_distance(self, phrase1: str | PhrasesTable, phrase2: str | PhrasesTable):
        embedding1 = compute_phrase_embedding(phrase1, tokenizer=self.tokenizer)
        embedding2 = compute_phrase_embedding(phrase2, tokenizer=self.tokenizer)
        return self.compute_distance(embedding1, embedding2)


class EuclideanDistancePhrasesRetriever(PhrasesRetriever):
    def get_distance_column(self, embedding: np.ndarray):
        return PhrasesTable.embedding.l2_distance(embedding).label("distance")

    def compute_distance(self, embedding1: np.ndarray, embedding2: np.ndarray) -> float:
        return np.sqrt(np.sum((embedding1 - embedding2) ** 2))


class CosineDistancePhrasesRetriever(PhrasesRetriever):
    def get_distance_column(self, embedding: np.ndarray):
        return PhrasesTable.embedding.cosine_distance(embedding).label("distance")

    def compute_distance(self, embedding1: np.ndarray, embedding2: np.ndarray) -> float:
        return 1 - np.dot(embedding1, embedding2) / (np.linalg.norm(embedding1) * np.linalg.norm(embedding2))


def get_phrase_retriever(tokenizer: Tokenizer) -> PhrasesRetriever:
    if Config.distance_calc == "L2":
        return EuclideanDistancePhrasesRetriever(tokenizer)
    elif Config.distance_calc == "Cosine":
        return CosineDistancePhrasesRetriever(tokenizer)
    raise ValueError(f"Invalid distance_calc type: '{Config.distance_calc}'")
