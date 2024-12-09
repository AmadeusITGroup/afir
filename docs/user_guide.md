# User Guide

This guide provides an overview of how to use the Fraud Investigation System.

## Submitting an Incident

1. Prepare the correct API call based on the desired input method (always with a JSON body). You can decide the names of the API endpoints in the `main_config` file.
   2. Textual input: the body must have the fields `id`, `timestamp`, `description`
   3. Win@proach retrieval: the body must have the field `id`

## Reviewing Investigation Results

1. Once an investigation is complete, a report will be generated in the `exports` folder.
2. The report will include:
    - Incident summary
    - Detected anomalies
    - Risk assessment
    - Recommended actions

## Configuring Plugins

1. Edit the `config/plugin_config.yaml` file to activate or deactivate plugins.
2. Restart the system for plugin changes to take effect.

For any additional assistance, please contact the system administrator or refer to the troubleshooting guide.