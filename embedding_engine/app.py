from fastapi import FastAPI
from pydantic import BaseModel

from embedding_engine.config import Config
from embedding_engine.database import Session
from embedding_engine.embedding.distances import get_phrase_retriever
from embedding_engine.embedding.tokenizer import get_tokenizer
from embedding_engine.embedding.utils import (
    compute_inter_phrase_distances,
    compute_phrase_embedding,
    save_phrase_embeddings,
)
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
phrase_retriever = get_phrase_retriever(tokenizer)

save_phrase_embeddings(tokenizer)
inter_phrase_distances = compute_inter_phrase_distances(phrase_retriever)

app = FastAPI()


class NearestRequest(BaseModel):
    phrase: str


class NearestResponse(BaseModel):
    phrase: str
    dist: float | None = None


class InterPhrasesDistances(BaseModel):
    distances: list[list[float]]


@app.post("/nearest")
def find_nearest(request: NearestRequest) -> NearestResponse:
    embedding = compute_phrase_embedding(request.phrase, tokenizer)
    nearest = phrase_retriever.get_nearest(embedding, Session())

    if nearest is None:
        return NearestResponse(phrase="**UNSUCCESSFUL**")

    phrase, dist = nearest
    return NearestResponse(phrase=phrase, dist=dist)


@app.get("/inter_phrase_distances")
def get_inter_phrase_distances():
    return InterPhrasesDistances(distances=inter_phrase_distances)
