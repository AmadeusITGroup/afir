import asyncio
import logging
from concurrent.futures import ProcessPoolExecutor

import yaml

from anomaly_detection import AnomalyDetectionModule
from api_call_generator import ApiCallGenerator
from export_results import ResultExporter
from feedback_loop import FeedbackLoop
from incident_input import IncidentInputInterface
from incident_understanding import IncidentUnderstandingModule
from log_retrieval import LogRetrievalEngine
from notifications import NotificationSystem, send_notification
from output_interface import OutputInterface
from plugin_system import PluginManager
from report_generation import ReportGenerationModule
from utils.error_handling import async_retry_with_backoff
from utils.llm_utils import RAG

logging.basicConfig(level=logging.INFO, format='%(asctime)s - %(levelname)s - %(message)s')
logger = logging.getLogger(__name__)


def load_config(config_file):
    with open(config_file, 'r') as f:
        return yaml.safe_load(f)


@async_retry_with_backoff(max_attempts=3, backoff_in_seconds=1)
async def process_incident(incident, modules):
    try:
        send_notification(incident['id'], 'processing', 'Started processing incident')

        understanding = await modules['understanding'].process(incident)
        logger.info(f"Incident {incident['id']} understanding complete")
        print(understanding)

        api_calls = await modules['api_call'].generate(understanding)
        logger.info(f"Generated {len(api_calls)} API calls for incident {incident['id']}")
        print(api_calls)

        if modules['log_retrieval'].config['use_ssh_tunnel']:
            logs = await modules['log_retrieval'].retrieve_with_tunnel(api_calls)
        else:
            logs = await modules['log_retrieval'].retrieve(api_calls)
        logger.info(f"Retrieved logs from {len(logs)} sources for incident {incident['id']}")
        print(logs)

        anomalies = await modules['anomaly_detection'].detect(logs, understanding)
        logger.info(f"Detected {len(anomalies)} anomalies for incident {incident['id']}")
        print(anomalies)

        # Use plugins for additional processing
        for plugin in modules['plugins'].get_active_plugins():
            plugin_result = await modules['plugins'].execute_plugin(plugin, incident, understanding, logs, anomalies)
            logger.info(f"Executed plugin {plugin} for incident {incident['id']} with result: {plugin_result}")

        report = await modules['report_generation'].generate(incident, understanding, logs, anomalies)
        logger.info(f"Generated investigation report for incident {incident['id']}")

        # await modules['output'].send(report, incident['id']) # TO BE IMPLEMENTED
        # logger.info(f"Sent investigation report for incident {incident['id']}")

        # Export results
        exporter = ResultExporter({'incident': incident, 'understanding': understanding, 'anomalies': anomalies})
        exporter.export_json(f"../exports/incident_{incident['id']}.json")
        exporter.export_csv(f"../exports/incident_{incident['id']}.csv")
        logger.info(f"Exported results for incident {incident['id']}")

        # Collect feedback (this would typically be done after human review) # TO BE IMPLEMENTED
        # await modules['feedback'].collect_feedback(incident['id'], {'anomalies': anomalies}, {'accuracy': 0.9})

        send_notification(incident['id'], 'completed', 'Incident processing completed')
        return report
    except Exception as e:
        logger.error(f"Error processing incident {incident['id']}: {str(e)}")
        send_notification(incident['id'], 'error', f'Error processing incident: {str(e)}')
        raise


async def main():
    main_config = load_config('../config/main_config.yaml')
    llm_config = load_config('../config/llm_config.yaml')

    # Initialize RAG
    use_rag = main_config['rag']['use_rag']
    if use_rag:
        rag = RAG(
            knowledge_base_path=main_config['knowledge_base']['path'],
            sentence_transformer_model=main_config['rag']['sentence_transformer_model'],
            max_retrieved_documents=main_config['rag']['max_retrieved_documents'],
            similarity_threshold=main_config['rag']['similarity_threshold']
        )
    else:
        rag = None

    # Initialize notification system
    notification_system = NotificationSystem()
    await notification_system.start_notification_server()

    modules = {
        'input': IncidentInputInterface(main_config['incident_input']),
        'understanding': IncidentUnderstandingModule(llm_config, rag),
        'api_call': ApiCallGenerator(main_config['log_sources'], llm_config),
        'log_retrieval': LogRetrievalEngine(main_config['log_sources']),
        'anomaly_detection': AnomalyDetectionModule(main_config['anomaly_detection'], llm_config, rag),
        'report_generation': ReportGenerationModule(main_config['report_generation'], llm_config),
        'output': OutputInterface(main_config['output_interface']),
        'plugins': PluginManager(main_config['plugin_dir']),
        'feedback': FeedbackLoop(llm_config)  # RAG feedback loop TO BE IMPLEMENTED
    }

    # Load the plugins
    modules['plugins'].load_plugins()

    # Start the incident input server
    await modules['input'].start_server()

    logger.info("Fraud Investigation System initialized. Waiting for incidents...")

    max_workers = main_config['performance']['max_workers']
    batch_size = main_config['performance']['batch_size']

    with ProcessPoolExecutor(max_workers=max_workers) as executor:
        while True:
            incidents = await modules['input'].get_incidents(batch_size)
            if not incidents:
                await asyncio.sleep(1)  # Avoid busy waiting
                continue

            tasks = [
                asyncio.create_task(process_incident(incident, modules))
                for incident in incidents
            ]

            results = await asyncio.gather(*tasks, return_exceptions=True)

            for incident, result in zip(incidents, results):
                if isinstance(result, Exception):
                    logger.error(f"Failed to process incident {incident['id']}: {str(result)}")
                else:
                    logger.info(f"Successfully processed incident {incident['id']}")

            # Process feedback periodically
            # await modules['feedback'].process_feedback() # TO IMPLEMENT


if __name__ == "__main__":
    asyncio.run(main())
