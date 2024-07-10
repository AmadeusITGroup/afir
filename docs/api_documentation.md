# API Documentation

The Fraud Investigation System exposes several API endpoints for integration with other systems.

## Incident Submission

Submit a new incident for investigation.

- **URL**: `/api/v1/incidents`
- **Method**: `POST`
- **Headers**:
    - Content-Type: application/json
    - Authorization: Bearer <your_token>
- **Body**:
  ```json
  {
    "id": "INC-12345",
    "timestamp": "2023-07-08T12:00:00Z",
    "description": "Unusual login activity detected",
    "severity": 8
  }
  ```
- **Response**:
    - Status: 201 Created
    - Body:
      ```json
      {
        "message": "Incident received",
        "incident_id": "INC-12345"
      }
      ```

## Get Investigation Status

Retrieve the status of an ongoing investigation.

- **URL**: `/api/v1/incidents/<incident_id>/status`
- **Method**: `GET`
- **Headers**:
    - Authorization: Bearer <your_token>
- **Response**:
    - Status: 200 OK
    - Body:
      ```json
      {
        "incident_id": "INC-12345",
        "status": "in_progress",
        "completion_percentage": 75
      }
      ```

## Get Investigation Report

Retrieve the final investigation report for a completed incident.

- **URL**: `/api/v1/incidents/<incident_id>/report`
- **Method**: `GET`
- **Headers**:
    - Authorization: Bearer <your_token>
- **Response**:
    - Status: 200 OK
    - Body: JSON object containing the full investigation report