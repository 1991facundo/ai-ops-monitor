from datetime import datetime
from pydantic import BaseModel
from typing import Optional, Dict, Any

class Event(BaseModel):
    source: str
    timestamp: datetime
    type: str
    severity: str
    message: str
    metadata: Optional[Dict[str, Any]] = None

class EventDB(Event):
    id: int
