from fastapi import FastAPI
from pydantic import BaseModel
from sqlalchemy import select

from embedding_engine.config import Config
from embedding_engine.database import Session
from embedding_engine.database.phrases_table import PhrasesTable
from embedding_engine.embedding.assign import compute_phrase_embedding, save_phrase_embeddings
from embedding_engine.embedding.distances import get_phrase_retriever
from embedding_engine.embedding.tokenizer import get_tokenizer
from embedding_engine.load_vectors import GoogleDriveVectorsDownloader

downloader = GoogleDriveVectorsDownloader(
    url=Config.drive_url,
    save_path=Config.save_path,
    extracted_path=Config.extracted_path,
    extracted_processed_path=Config.extracted_processed_path,
)
downloader.download()
downloader.extract()
downloader.insert_into_db()

tokenizer = get_tokenizer()
save_phrase_embeddings(tokenizer)
euclidean_distance = get_phrase_retriever(tokenizer)

inter_phrase_distances: list[list[float]] = []
with Session() as session:
    statement = select(PhrasesTable)
    scalars = list(session.scalars(statement))
    for p1_ix, phrase1 in enumerate(scalars):
        inter_phrase_distances.append([])
        for p2_ix, phrase2 in enumerate(scalars):
            inter_phrase_distances[p1_ix].append(euclidean_distance.get_distance(phrase1.phrase, phrase2.phrase))

app = FastAPI()


class NearestRequest(BaseModel):
    phrase: str


class NearestResponse(BaseModel):
    phrase: str
    dist: float | None = None


@app.post("/nearest")
def find_nearest(request: NearestRequest) -> NearestResponse:
    embedding = compute_phrase_embedding(request.phrase, tokenizer)
    nearest = euclidean_distance.get_nearest(embedding, Session())

    if nearest is None:
        return NearestResponse(phrase="**UNSUCCESSFUL**")

    phrase, dist = nearest
    return NearestResponse(phrase=phrase, dist=dist)
