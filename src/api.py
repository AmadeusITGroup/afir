from flask import Flask, request, jsonify
from src.main import process_incident
import asyncio

app = Flask(__name__)


@app.route('/api/submit_incident', methods=['POST'])
def submit_incident():
    incident_data = request.json
    if not incident_data:
        return jsonify({"error": "No data provided"}), 400

    required_fields = ['id', 'timestamp', 'description', 'severity']
    if not all(field in incident_data for field in required_fields):
        return jsonify({"error": "Missing required fields"}), 400

    # Asynchronously process the incident
    asyncio.create_task(process_incident(incident_data))

    return jsonify({"message": "Incident submitted successfully", "incident_id": incident_data['id']}), 202


@app.route('/api/get_result/<incident_id>', methods=['GET'])
def get_result(incident_id):
    # In a real implementation, this would query a database or cache for the result
    # For this example, we'll return a mock result
    mock_result = {
        "incident_id": incident_id,
        "status": "completed",
        "anomalies_detected": 2,
        "risk_score": 0.8
    }
    return jsonify(mock_result)


@app.route('/api/system_status', methods=['GET'])
def system_status():
    # In a real implementation, this would check various system components
    status = {
        "status": "operational",
        "incident_queue_size": 5,
        "processing_rate": "10 incidents/minute",
        "llm_status": "online"
    }
    return jsonify(status)


if __name__ == '__main__':
    app.run(debug=True, port=5000)

# Example usage (you can put this in a separate file or use a tool like curl or Postman to test)
import requests


def test_api():
    # Submit an incident
    incident_data = {
        "id": "INC-67890",
        "timestamp": "2023-07-06T14:30:00",
        "description": "Suspicious fund transfer detected",
        "severity": 9
    }
    response = requests.post('http://localhost:5000/api/submit_incident', json=incident_data)
    print("Submit incident response:", response.json())

    # Get result for an incident
    incident_id = "INC-67890"
    response = requests.get(f'http://localhost:5000/api/get_result/{incident_id}')
    print("Get result response:", response.json())

    # Check system status
    response = requests.get('http://localhost:5000/api/system_status')
    print("System status response:", response.json())


if __name__ == "__main__":
    test_api()
