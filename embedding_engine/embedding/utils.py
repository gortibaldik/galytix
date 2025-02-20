from logging import getLogger

from sqlalchemy import insert, select

from embedding_engine.config import Config
from embedding_engine.database import PhrasesTable, Session, check_table_empty
from embedding_engine.embedding.compute import compute_phrase_embedding
from embedding_engine.embedding.distances import PhrasesRetriever
from embedding_engine.embedding.tokenizer import Tokenizer

logger = getLogger(__name__)


def save_phrase_embeddings(tokenizer: Tokenizer):
    if not check_table_empty(PhrasesTable):
        logger.warning("PhrasesTable already initialized, skipping")
        return

    with open(Config.phrases_path, "r", encoding="unicode_escape") as f:
        for i, line in enumerate(f):
            if i == 0:
                continue
            phrase = line.strip()
            embedding = compute_phrase_embedding(phrase, tokenizer)

            with Session() as session:
                statement = insert(PhrasesTable).values(phrase=phrase, embedding=embedding)
                session.execute(statement)
                session.commit()


def compute_inter_phrase_distances(phrases_retriever: PhrasesRetriever):
    inter_phrase_distances: list[list[float]] = []
    with Session() as session:
        statement = select(PhrasesTable)
        scalars = list(session.scalars(statement))
        for p1_ix, phrase1 in enumerate(scalars):
            inter_phrase_distances.append([])
            for p2_ix, phrase2 in enumerate(scalars):
                inter_phrase_distances[p1_ix].append(phrases_retriever.get_distance(phrase1.phrase, phrase2.phrase))

    return inter_phrase_distances
