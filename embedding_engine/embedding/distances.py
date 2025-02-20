import numpy as np
from sqlalchemy import select
from sqlalchemy.orm import Session

from embedding_engine.database import PhrasesTable
from embedding_engine.embedding.assign import compute_phrase_embedding
from embedding_engine.embedding.tokenizer import Tokenizer


class EuclideanDistancePhrasesRetriever:
    def __init__(self, tokenizer: Tokenizer):
        self.tokenizer = tokenizer

    def get_nearest(self, embedding: np.ndarray, session: Session, limit: int = 1) -> None | tuple[str, float]:
        distance_column = PhrasesTable.embedding.l2_distance(embedding).label("distance")
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

        return np.sqrt(np.sum((embedding1 - embedding2) ** 2))
