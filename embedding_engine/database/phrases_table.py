from embedding_engine.database.base import Base
from pgvector.sqlalchemy import Vector
from sqlalchemy.orm import mapped_column, Mapped

class PhrasesTable(Base):
    __tablename__ = "phrases"

    phrase: Mapped[str] = mapped_column(primary_key=True)
    embedding = mapped_column(Vector(300))

    def __repr__(self):
        return f"<Phrases(phrase='{self.phrase}', embedding_len='{len(self.embedding)}')>"