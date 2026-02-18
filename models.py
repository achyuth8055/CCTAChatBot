import uuid
from datetime import datetime
from enum import Enum as PyEnum

from sqlalchemy import create_engine, Column, String, Integer, Text, DateTime, Enum
from sqlalchemy.ext.declarative import declarative_base
from sqlalchemy.orm import sessionmaker

DATABASE_URL = "sqlite:///documents.db"

engine = create_engine(DATABASE_URL, echo=False)
SessionLocal = sessionmaker(autocommit=False, autoflush=False, bind=engine)
Base = declarative_base()


class DocumentStatus(str, PyEnum):
    PENDING = "pending"
    PROCESSING = "processing"
    TRAINED = "trained"
    FAILED = "failed"
    DELETING = "deleting"


class Document(Base):
    __tablename__ = "documents"

    id = Column(String(36), primary_key=True, default=lambda: str(uuid.uuid4()))
    chatbot_id = Column(String(36), nullable=False, index=True)
    filename = Column(String(255), nullable=False)
    storage_path = Column(String(512), nullable=False)
    file_size = Column(Integer, nullable=True)
    mime_type = Column(String(100), default="application/pdf")
    status = Column(
        Enum(DocumentStatus),
        default=DocumentStatus.PENDING,
        nullable=False
    )
    error_message = Column(Text, nullable=True)
    chunk_count = Column(Integer, default=0)
    created_at = Column(DateTime, default=datetime.utcnow)
    updated_at = Column(DateTime, default=datetime.utcnow, onupdate=datetime.utcnow)

    def to_dict(self):
        return {
            "id": self.id,
            "chatbot_id": self.chatbot_id,
            "filename": self.filename,
            "file_size": self.file_size,
            "mime_type": self.mime_type,
            "status": self.status.value if isinstance(self.status, DocumentStatus) else self.status,
            "error_message": self.error_message,
            "chunk_count": self.chunk_count,
            "created_at": self.created_at.isoformat() if self.created_at else None,
            "updated_at": self.updated_at.isoformat() if self.updated_at else None,
        }

    def __repr__(self):
        return f"<Document {self.filename} ({self.status})>"


def init_db():
    Base.metadata.create_all(bind=engine)
    print("[DB] Database initialized successfully.")


def get_db():
    db = SessionLocal()
    try:
        yield db
    finally:
        db.close()
