import os
import json
import openai
from typing import List

from fastapi import Depends, FastAPI, HTTPException
from fastapi.middleware.cors import CORSMiddleware
from sqlalchemy import text
from sqlalchemy.orm import Session

from app.database import EventDB as EventDBModel, SessionLocal, init_db
from app.models import Event
from app.schemas import EventDB

from qdrant_client import QdrantClient
from qdrant_client.http.models import Distance, VectorParams


app = FastAPI(title="AI Ops Monitor API", version="1.0")

# CORS abierto por ahora (AFINARLO LUEGO)
app.add_middleware(
    CORSMiddleware,
    allow_origins=["*"],
    allow_methods=["*"],
    allow_headers=["*"],
)


# OpenAI

OPENAI_API_KEY = os.getenv("OPENAI_API_KEY")
openai.api_key = OPENAI_API_KEY


def get_db():
    db = SessionLocal()
    try:
        yield db
    finally:
        db.close()


@app.get("/health")
async def health():
    return {"status": "ok"}


@app.post("/events")
async def receive_event(event: Event, db: Session = Depends(get_db)):
    # 1) Guardar en la base SQLite
    metadata = event.metadata or {}
    data_payload = {**metadata, "type": event.type, "severity": event.severity}

    db_event = EventDBModel(
        source=event.source,
        level=event.severity,
        message=event.message,
        data=json.dumps(data_payload),
        timestamp=event.timestamp.isoformat(),
    )
    db.add(db_event)
    db.commit()
    db.refresh(db_event)

    # 2) Generar embedding con OpenAI
    vector = None
    try:
        embed_response = openai.Embedding.create(
            input=event.message, model="text-embedding-ada-002"
        )
        vector = embed_response["data"][0]["embedding"]
    except Exception as e:
        # En desarrollo, lo logueamos y seguimos sin romper la API
        print(f"[Embedding error] {e}")

    # 3) Indexar en Qdrant si el embedding se generó bien
    if vector is not None:
        try:
            qdrant_client.upsert(
                collection_name=COLLECTION_NAME,
                points=[
                    {
                        "id": db_event.id,
                        "vector": vector,
                        "payload": {
                            "source": event.source,
                            "type": event.type,
                            "severity": event.severity,
                            "timestamp": event.timestamp.isoformat(),
                            "metadata": event.metadata or {},
                        },
                    }
                ],
            )
        except Exception as e:
            print(f"[Qdrant upsert error] {e}")

    return {"status": "stored", "id": db_event.id, "event": event}


@app.post("/events/similar", response_model=List[Event])
async def find_similar_events(query_event: Event, db: Session = Depends(get_db)):
    try:
        # 1) Generar embedding del evento de consulta
        embed_response = openai.Embedding.create(
            input=query_event.message, model="text-embedding-ada-002"
        )
        query_vector = embed_response["data"][0]["embedding"]
    except Exception as e:
        print(f"[Embedding error in /events/similar] {e}")
        raise HTTPException(
            status_code=500, detail="Error generating embedding for similarity search"
        )

    # 2) Buscar en Qdrant los más similares
    search_result = qdrant_client.search(
        collection_name=COLLECTION_NAME, query_vector=query_vector, limit=5
    )

    similar_ids = [int(point.id) for point in search_result]
    if not similar_ids:
        return []

    # 3) Leer esos eventos desde la DB
    rows = db.query(EventDBModel).filter(EventDBModel.id.in_(similar_ids)).all()

    results: List[Event] = []
    for row in rows:
        metadata = {}
        if row.data:
            try:
                metadata = json.loads(row.data)
            except Exception:
                metadata = {}

        event_type = metadata.pop("type", None)
        severity = metadata.pop("severity", row.level)

        results.append(
            Event(
                id=str(row.id),
                source=row.source,
                type=event_type or "",
                severity=severity,
                message=row.message,
                timestamp=row.timestamp,
                metadata=metadata,
            )
        )

    return results


@app.get("/events", response_model=List[EventDB])
async def list_events(
    level: str | None = None,
    source: str | None = None,
    limit: int = 100,
    db: Session = Depends(get_db),
):
    base_query = "SELECT id, source, timestamp, level, message, data FROM events"
    filters = []
    params: dict = {}

    if level:
        filters.append("level = :level")
        params["level"] = level

    if source:
        filters.append("source = :source")
        params["source"] = source

    if filters:
        base_query += " WHERE " + " AND ".join(filters)

    base_query += " ORDER BY timestamp DESC LIMIT :limit"
    params["limit"] = limit

    result = db.execute(text(base_query), params)
    rows = result.fetchall()

    events: list[dict] = []
    for row in rows:
        events.append(
            {
                "id": row[0],
                "source": row[1],
                "timestamp": row[2],
                "level": row[3],
                "message": row[4],
                "data": row[5],
            }
        )

    return events


# Cliente Qdrant dentro del entorno Docker
qdrant_client = QdrantClient(host="qdrant", port=6333)

COLLECTION_NAME = "events_vectors"
VECTOR_SIZE = int(os.getenv("VECTOR_DIMENSION", "1536"))


# Inicializar colección si no existe
def init_vector_collection():
    qdrant_client.recreate_collection(
        collection_name=COLLECTION_NAME,
        vectors_config=VectorParams(size=VECTOR_SIZE, distance=Distance.COSINE),
    )


init_db()
init_vector_collection()
