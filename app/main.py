import logging
import os
import json
import openai
from typing import List, Optional

from fastapi import Depends, FastAPI, HTTPException
from fastapi.middleware.cors import CORSMiddleware
from sqlalchemy import text
from sqlalchemy.orm import Session

from app.database import EventDB as EventDBModel, SessionLocal, init_db
from app.models import Event
from app.schemas import EventDB

from qdrant_client import QdrantClient
from qdrant_client.http.models import Distance, VectorParams


logger = logging.getLogger(__name__)


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
    source: Optional[str] = None,
    severity: Optional[str] = None,
    limit: int = 100,
    db: Session = Depends(get_db),
):
    """
    Lista eventos usando el modelo ORM EventDBModel.
    Filtra opcionalmente por source y severity.
    Reconstruye type y metadata a partir del JSON guardado en data.
    """

    try:
        # Construimos la query con SQLAlchemy, sin SQL crudo
        query = db.query(EventDBModel)

        if source is not None:
            query = query.filter(EventDBModel.source == source)

        if severity is not None:
            # En el modelo ORM el campo se llama "level" (severity lógico)
            query = query.filter(EventDBModel.level == severity)

        # Ordenar por timestamp descendente y limitar
        query = query.order_by(EventDBModel.timestamp.desc()).limit(limit)
        rows = query.all()

        events: list[dict] = []

        for row in rows:
            # row.data es el JSON string que guardamos en el POST /events
            event_type = ""
            metadata_dict = {}

            if row.data:
                try:
                    json_data = json.loads(row.data)
                    # En el POST metemos "type" y el resto del metadata adentro
                    event_type = json_data.get("type", "") or ""
                    # Si guardaste metadata separado, lo tomamos; si no, usamos el resto del dict
                    metadata_dict = json_data.get("metadata", {})
                    if not metadata_dict:
                        # fallback: todo excepto type/severity
                        metadata_dict = {
                            k: v
                            for k, v in json_data.items()
                            if k not in ("type", "severity")
                        }
                except Exception:
                    metadata_dict = {}

            events.append(
                {
                    "id": row.id,
                    "source": row.source,
                    "timestamp": row.timestamp,
                    "type": event_type,
                    "severity": row.level,  # "level" en la DB, "severity" en el modelo lógico
                    "message": row.message,
                    "metadata": metadata_dict,
                }
            )

        return events

    except Exception:
        logger.exception("Error listing events")
        raise



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
