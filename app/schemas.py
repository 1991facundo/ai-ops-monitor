from datetime import datetime
from pydantic import BaseModel


class Event(BaseModel):
    source: str
    timestamp: datetime
    level: str
    message: str
    data: dict | None = None


class EventDB(Event):
    id: int
