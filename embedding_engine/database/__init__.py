from logging import getLogger

import backoff
import psycopg2
from sqlalchemy import create_engine, text
from sqlalchemy.orm import sessionmaker

from embedding_engine.config import Config
from embedding_engine.database.base import Base
from embedding_engine.database.phrases_table import PhrasesTable
from embedding_engine.database.vectors_table import VectorsTable

logger = getLogger(__name__)
engine = create_engine(Config.db_connection_str)


@backoff.on_exception(wait_gen=backoff.expo, max_time=120, exception=psycopg2.OperationalError)
def initialize_database():
    # if the database schema is not initialized, initialize
    with engine.begin() as connection:
        connection.execute(text("CREATE extension IF NOT EXISTS vector;"))
        if not engine.dialect.has_table(connection, VectorsTable.__tablename__):
            logger.warning("CREATING DATABASE")
            Base.metadata.create_all(connection)
    logger.warning("CREATED DATABASE")


def check_table_empty(table: type[Base]):
    with Session() as session:
        value = session.query(table).first()
        return value is None


initialize_database()
Session = sessionmaker(bind=engine)

__all__ = ["Base", "VectorsTable", "PhrasesTable", "engine"]
