# User Guide

This guide provides an overview of how to use the Fraud Investigation System.

## Submitting an Incident

1. Prepare the incident data in JSON format.
2. Use the API endpoint `/api/v1/incidents` to submit the incident.
3. Note the returned incident ID for future reference.

## Monitoring Investigation Progress

1. Use the API endpoint `/api/v1/incidents/<incident_id>/status` to check the status of an investigation.
2. The system will provide updates on the investigation progress.

## Reviewing Investigation Results

1. Once an investigation is complete, use the API endpoint `/api/v1/incidents/<incident_id>/report` to retrieve the full report.
2. The report will include:
    - Incident summary
    - Detected anomalies
    - Risk assessment
    - Recommended actions

## Using the Dashboard

1. Access the dashboard at `http://your-server/dashboard`
2. The dashboard provides:
    - Overview of recent incidents
    - Anomaly trends
    - System performance metrics

## Configuring Plugins

1. Edit the `config/plugin_config.yaml` file to activate or deactivate plugins.
2. Restart the system for plugin changes to take effect.

## Providing Feedback

1. After reviewing an investigation report, use the feedback mechanism to rate the accuracy and usefulness of the results.
2. This feedback helps improve the system's performance over time.

For any additional assistance, please contact the system administrator or refer to the troubleshooting guide.