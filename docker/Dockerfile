FROM python:3.12

WORKDIR /usr/src/app

COPY embedding_engine/ ./embedding_engine
RUN ls
RUN ls
RUN pip install -r ./embedding_engine/requirements.txt
RUN pip install -e ./embedding_engine

CMD ["uvicorn", "embedding_engine.app:app", "--host", "0.0.0.0", "--port", "5000"]
