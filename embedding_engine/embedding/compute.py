from logging import getLogger

import numpy as np
from sqlalchemy import select

from embedding_engine.database import PhrasesTable, Session, SessionType, VectorsTable
from embedding_engine.embedding.tokenizer import Tokenizer

logger = getLogger(__name__)


def find_word_vector(word: str, session: SessionType):
    statement = select(VectorsTable).where(VectorsTable.word == word)
    result = session.execute(statement)
    words = list(result.scalars())

    if not words:
        return None

    if len(words) > 1:
        raise ValueError(f"Found many entries for '{word}' ({words})")

    return words[0].embedding


def compute_phrase_embedding(phrase: str | PhrasesTable, tokenizer: Tokenizer):
    if isinstance(phrase, PhrasesTable):
        return phrase.embedding

    words = tokenizer.tokenize(phrase)

    with Session() as session:
        statement = select(PhrasesTable).where(PhrasesTable.phrase == phrase)
        result = list(session.scalars(statement))

        if result:
            return result[0].embedding

        embeddings = []
        for word in words:
            if (embedding := find_word_vector(word, session)) is not None:
                embeddings.append(embedding)

    if not embeddings:
        raise NotImplementedError("No embeddings found for the sequence")

    return np.mean(embeddings, axis=0)
