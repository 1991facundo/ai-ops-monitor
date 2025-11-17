import os
import json
import openai

from fastapi import FastAPI, Depends, HTTPException
from fastapi.middleware.cors import CORSMiddleware
from app.models import Event

from app.database import EventDB, Base, engine, SessionLocal
from app.database import init_db
from sqlalchemy.orm import Session

from qdrant_client import QdrantClient
from qdrant_client.http.models import Distance, VectorParams

from typing import List



app = FastAPI(
    title="AI Ops Monitor API",
    version="1.0"
)

# CORS abierto por ahora (AFINARLO LUEGO)
app.add_middleware(
    CORSMiddleware,
    allow_origins=["*"],
    allow_methods=["*"],
    allow_headers=["*"],
)


#OpenAI

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
    db_event = EventDB(
        source=event.source,
        type=event.type,
        severity=event.severity,
        message=event.message,
        metadata_json=json.dumps(event.metadata),  # ⬅️ nombre correcto de columna
        timestamp=event.timestamp
    )
    db.add(db_event)
    db.commit()
    db.refresh(db_event)

    # 2) Generar embedding con OpenAI
    vector = None
    try:
        embed_response = openai.Embedding.create(
            input=event.message,
            model="text-embedding-ada-002"
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
                points=[{
                    "id": db_event.id,
                    "vector": vector,
                    "payload": {
                        "source": event.source,
                        "type": event.type,
                        "severity": event.severity,
                        "timestamp": event.timestamp.isoformat(),
                        "metadata": event.metadata or {}
                    }
                }]
            )
        except Exception as e:
            print(f"[Qdrant upsert error] {e}")

    return {
        "status": "stored",
        "id": db_event.id,
        "event": event
    }

@app.post("/events/similar", response_model=List[Event])
async def find_similar_events(query_event: Event, db: Session = Depends(get_db)):
    try:
        # 1) Generar embedding del evento de consulta
        embed_response = openai.Embedding.create(
            input=query_event.message,
            model="text-embedding-ada-002"
        )
        query_vector = embed_response["data"][0]["embedding"]
    except Exception as e:
        print(f"[Embedding error in /events/similar] {e}")
        raise HTTPException(status_code=500, detail="Error generating embedding for similarity search")


    # 2) Buscar en Qdrant los más similares
    search_result = qdrant_client.search(
        collection_name=COLLECTION_NAME,
        query_vector=query_vector,
        limit=5
    )

    similar_ids = [int(point.id) for point in search_result]
    if not similar_ids:
        return []

    # 3) Leer esos eventos desde la DB
    rows = (
        db.query(EventDB)
        .filter(EventDB.id.in_(similar_ids))
        .all()
    )

    results: List[Event] = []
    for row in rows:
        metadata = {}
        if row.metadata_json:
            try:
                metadata = json.loads(row.metadata_json)
            except Exception:
                metadata = {}

        results.append(
            Event(
                id=str(row.id),
                source=row.source,
                type=row.type,
                severity=row.severity,
                message=row.message,
                timestamp=row.timestamp,
                metadata=metadata,
            )
        )

    return results


# Cliente Qdrant dentro del entorno Docker
qdrant_client = QdrantClient(
    host="qdrant",
    port=6333
)

COLLECTION_NAME = "events_vectors"
VECTOR_SIZE = int(os.getenv("VECTOR_DIMENSION", "1536"))

# Inicializar colección si no existe
def init_vector_collection():
    qdrant_client.recreate_collection(
        collection_name=COLLECTION_NAME,
        vectors_config=VectorParams(
            size=VECTOR_SIZE,
            distance=Distance.COSINE
        )
    )

init_db()
init_vector_collection()