from sqlalchemy import create_engine, Column, Integer, String, Text, text
from sqlalchemy.orm import declarative_base, sessionmaker
from datetime import datetime

DATABASE_URL = "sqlite:///./events.db"

engine = create_engine(DATABASE_URL, connect_args={"check_same_thread": False})

SessionLocal = sessionmaker(bind=engine)

Base = declarative_base()


class EventDB(Base):
    __tablename__ = "events"

    id = Column(Integer, primary_key=True, index=True)
    source = Column(String, index=True)
    level = Column(String, index=True)
    message = Column(Text)
    timestamp = Column(Text, default=datetime.utcnow)
    data = Column(Text)


def init_db():
    with engine.connect() as conn:
        conn.execute(
            text(
                """
        CREATE TABLE IF NOT EXISTS events (
            id INTEGER PRIMARY KEY,
            source TEXT,
            timestamp TEXT,
            level TEXT,
            message TEXT,
            data TEXT
        );
        """
            )
        )
