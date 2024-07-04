from flask import Flask, render_template, request, jsonify
from src.main import process_incident
import asyncio

app = Flask(__name__)


@app.route('/')
def index():
    return render_template('index.html')


@app.route('/incidents')
def get_incidents():
    # This would be replaced with actual logic to fetch recent incidents
    incidents = [
        {'id': 'INC-12345', 'timestamp': '2023-07-05T10:30:00', 'status': 'Processing'},
        {'id': 'INC-67890', 'timestamp': '2023-07-05T11:45:00', 'status': 'Completed'}
    ]
    return jsonify(incidents)


@app.route('/incident/<incident_id>')
def get_incident_details(incident_id):
    # This would be replaced with actual logic to fetch incident details
    incident = {
        'id': incident_id,
        'timestamp': '2023-07-05T10:30:00',
        'description': 'Unusual login activity',
        'status': 'Processing',
        'anomalies': [
            {'description': 'Multiple failed logins', 'confidence': 0.8}
        ]
    }
    return jsonify(incident)


@app.route('/manual_process', methods=['POST'])
def manual_process():
    incident_data = request.json
    asyncio.run(process_incident(incident_data, app.config['modules']))
    return jsonify({'status': 'Processing started'})


if __name__ == '__main__':
    app.run(debug=True)
