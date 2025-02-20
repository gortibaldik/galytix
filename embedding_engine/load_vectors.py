import re
from logging import getLogger
from pathlib import Path

import gdown
from gensim.models import KeyedVectors

from embedding_engine.database import check_table_empty, engine
from embedding_engine.database.vectors_table import VectorsTable

logger = getLogger(__name__)


class GoogleDriveVectorsDownloader:
    """Download and preprocess files from Google Drive.

    Preprocessing steps include:
    1. reloading from binary format to the .csv format
    2. reloading from .csv format to .csv format that can be digested by psycopg2 to insert into postgresql database
    3. inserting word vectors into the postgresql database

    Args:
        url (str): google drive url for download of the vector file
        save_path (str): path where the downloaded google drive archive should be saved
        extracted_path (str): path where the raw extracted data will be saved
        extracted_processed_path (str): path where the result of the first preprocessing step is saved
        vectors_limit (int): number of vectors to load from the google drive archive
    """

    def __init__(
        self,
        url: str,
        save_path: str,
        extracted_path: str,
        extracted_processed_path: str,
        vectors_limit: int = 1_000_000,
    ):
        m = re.match(r"https://drive.google.com/file/d/([^/]+).*", url)
        if m is None:
            raise ValueError("Unexpected URL format! (url)")
        self.file_id = m.group(1)
        self.url = f"https://drive.google.com/ucuc?export=download&id={self.file_id}"
        self.save_path = save_path
        self.extracted_path = extracted_path
        self.extracted_processed_path = extracted_processed_path
        self.vectors_limit = vectors_limit

    def download(self):
        """Download data from the google drive archive.

        No-op if self.save_path exists.
        """
        if Path(self.save_path).exists():
            return

        try:
            logger.warning(f"GOING TO USE FILE ID: '{self.file_id}' for downloading from google drive")
            gdown.download(url=self.url, output=self.save_path, fuzzy=True)
        except gdown.exceptions.FileURLRetrievalError as e:
            raise ValueError(
                f"Please, download the file '{self.url}' from browser and add move it to '{self.save_path}'"
            ) from e

    def extract(self):
        """Extract data from downloaded google drive archive and preprocess them to csv format."""
        if not Path(self.save_path).exists():
            raise FileNotFoundError(f"'{self.save_path}' should exist before extraction. Did you run self.download() ?")

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
                    print(f'"{words[0]}"' + ',"[' + ",".join(words[1:]) + ']"', file=f2)
            logger.warning("PROCESSING FINISHED")

    def insert_into_db(self):
        """Insert extracted data to the postgresql database."""
        if not check_table_empty(VectorsTable):
            logger.warning("Word vectors already put into database, SKIPPING")
            return

        if not Path(self.extracted_processed_path).exists():
            raise FileNotFoundError(
                f"'{self.extracted_processed_path}' should exist before insertion into db. Did you run self.extract() ?"
            )

        logger.warning("INSERTING word vectors DATA INTO DATABASE")

        connection = engine.raw_connection()
        cursor = connection.cursor()

        with open(self.extracted_processed_path, "r") as f:
            cmd = "COPY vectors(word, embedding) FROM STDIN WITH (FORMAT CSV, HEADER FALSE)"
            cursor.copy_expert(cmd, f)

        connection.commit()
        logger.warning("word vectors INSERTED")
