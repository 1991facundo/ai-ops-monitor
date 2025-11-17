from pydantic import BaseModel, Field
from typing import Optional, Dict, Any
from datetime import datetime

class Event(BaseModel):
    id: Optional[str] = Field(
        None, description="Event ID. If not provided, one will be generated."
    )
    
    source: str = Field(
        ..., description="System that generated the event (api, n8n, qdrant, worker, llm, client, etc.)"
    )
    
    type: str = Field(
        ..., description="Type of event: log, metric, alert, decision, error, request, response, fallback"
    )
    
    severity: Optional[str] = Field(
        None, description="debug, info, warning, error, critical"
    )
    
    message: str = Field(
        ..., description="Human-readable message describing the event"
    )
    
    timestamp: datetime = Field(
        default_factory=datetime.utcnow,
        description="Event timestamp"
    )
    
    metadata: Optional[Dict[str, Any]] = Field(
        default_factory=dict,
        description="Additional structured data"
    )
