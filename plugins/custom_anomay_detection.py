# Example plugin: Custom Anomaly Detection
# plugins/custom_anomaly_detection.py
def custom_anomaly_detection(logs):
    # Implement custom anomaly detection logic
    anomalies = []
    for log in logs:
        if 'error' in log.lower():
            anomalies.append({'description': 'Error detected', 'confidence': 0.9})
    return anomalies


def register_plugin():
    return {
        'name': 'custom_anomaly_detection',
        'description': 'Custom anomaly detection plugin',
        'execute': custom_anomaly_detection
    }


