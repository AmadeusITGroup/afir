from pydantic import BaseModel, Field, validator
from typing import Dict, Any
from datetime import datetime
import re

class Incident(BaseModel):
    id: str = Field(..., min_length=1, max_length=50)
    timestamp: datetime
    description: str = Field(..., min_length=10, max_length=1000)
    source: str = Field(..., min_length=1, max_length=100)
    severity: int = Field(..., ge=1, le=10)
    additional_data: Dict[str, Any] = {}

    @validator('id')
    def id_alphanumeric(cls, v):
        if not re.match(r'^[a-zA-Z0-9_-]+$', v):
            raise ValueError('ID must be alphanumeric')
        return v

    @validator('timestamp')
    def timestamp_not_in_future(cls, v):
        if v > datetime.now():
            raise ValueError('Timestamp cannot be in the future')
        return v

def validate_incident(incident_data: dict) -> bool:
    try:
        Incident(**incident_data)
        return True
    except ValueError as e:
        print(f"Validation error: {str(e)}")
        return False

# Example usage
if __name__ == "__main__":
    valid_incident = {
        "id": "INC-12345",
        "timestamp": "2023-07-05T10:30:00",
        "description": "Unusual login activity detected from multiple IP addresses.",
        "source": "IDS",
        "severity": 8,
        "additional_data": {
            "ip_addresses": ["192.168.1.100", "10.0.0.5"],
            "user_id": "user123"
        }
    }

    invalid_incident = {
        "id": "INC 12345",  # Invalid: contains space
        "timestamp": "2025-01-01T00:00:00",  # Invalid: future date
        "description": "Short",  # Invalid: too short
        "source": "IDS",
        "severity": 11,  # Invalid: out of range
        "additional_data": {}
    }

    print("Valid incident:", validate_incident(valid_incident))
    print("Invalid incident:", validate_incident(invalid_incident))