from fastapi import FastAPI, Depends
from fastapi.middleware.cors import CORSMiddleware
from app.models import Event

from app.database import EventDB, Base, engine, SessionLocal
from sqlalchemy.orm import Session
import json

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

Base.metadata.create_all(bind=engine)

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
    db_event = EventDB(
        source=event.source,
        type=event.type,
        severity=event.severity,
        message=event.message,
        metadata=json.dumps(event.metadata),
        timestamp=event.timestamp
    )
    db.add(db_event)
    db.commit()
    db.refresh(db_event)

    return {
        "status": "stored",
        "id": db_event.id,
        "event": event
    }
