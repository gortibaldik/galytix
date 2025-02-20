from pathlib import Path
import re
import gdown
from gensim.models import KeyedVectors
from embedding_engine.database import check_table_empty, engine
from sqlalchemy.orm import Session
from logging import getLogger

from embedding_engine.database.vectors_table import VectorsTable

logger = getLogger(__name__)

class GoogleDriveVectorsDownloader:
    def __init__(
        self,
        url: str,
        save_path: str,
        extracted_path: str,
        extracted_processed_path: str,
        vectors_limit: int = 1_000_000
    ):
        m = re.match(r"https://drive.google.com/file/d/([^/]+).*", url)
        if m is None:
            raise ValueError(f"Unexpected URL format! (url)")
        self.file_id = m.group(1)
        self.url = f"https://drive.google.com/ucuc?export=download&id={self.file_id}"
        self.save_path = save_path
        self.extracted_path = extracted_path
        self.extracted_processed_path = extracted_processed_path
        self.vectors_limit = vectors_limit

    def download(self):
        if Path(self.save_path).exists():
            return
        
        try:
            logger.warning(f"GOING TO USE FILE ID: '{self.file_id}' for downloading from google drive")
            gdown.download(url=self.url, output=self.save_path, fuzzy=True)
        except gdown.exceptions.FileURLRetrievalError as e:
            raise ValueError(f"Please, download the file '{self.url}' from browser and add move it to '{self.save_path}'") from e
        
    def extract(self):
        if not Path(self.extracted_path).exists():
            wv = KeyedVectors.load_word2vec_format(self.save_path, binary=True, limit=self.vectors_limit)
            wv.save_word2vec_format(self.extracted_path)

        if not Path(self.extracted_processed_path).exists():
            logger.warning("PROCESSING EXTRACTED FILE!")
            with open(self.extracted_path, "r") as f, open(self.extracted_processed_path, "w") as f2:
                for i, line in enumerate(f):
                    if i == 0:
                        continue
                    
                    line = line.rstrip()
                    words = line.split(" ")
                    print(f"\"{words[0]}\"" + ",\"[" + ",".join(words[1:]) + "]\"", file=f2)
            logger.warning("PROCESSING FINISHED")

    def insert_into_db(self):            
        if not check_table_empty(VectorsTable):
            logger.warning("Word vectors already put into database, SKIPPING")
            return


        connection = engine.raw_connection()
        cursor = connection.cursor()

        with open(self.extracted_processed_path, "r") as f:
            cmd = 'COPY vectors(word, embedding) FROM STDIN WITH (FORMAT CSV, HEADER FALSE)'
            cursor.copy_expert(cmd, f)
        
        connection.commit()