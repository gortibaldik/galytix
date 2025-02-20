from embedding_engine.database.base import Base
from pgvector.sqlalchemy import Vector
from sqlalchemy.orm import mapped_column, Mapped

class VectorsTable(Base):
    __tablename__ = "vectors"

    word: Mapped[str] = mapped_column(primary_key=True)
    embedding = mapped_column(Vector(300))

    def __repr__(self):
        return f"<Vectors(word='{self.word}', embedding_len='{len(self.embedding)}')>"