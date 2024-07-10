import asyncio
import json

from elasticsearch import AsyncElasticsearch
import aiohttp
import logging
from src.utils.error_handling import async_retry_with_backoff

logger = logging.getLogger(__name__)


def build_elasticsearch_query(api_call):
    query = {
        "query": {
            "bool": {
                "must": [
                    {"range": {"timestamp": {
                        "gte": api_call['query_parameters'].get('start_time'),
                        "lte": api_call['query_parameters'].get('end_time')
                    }}},
                    {"query_string": {"query": api_call['query_parameters'].get('search_terms', '*')}}
                ]
            }
        }
    }
    return query


class LogRetrievalEngine:
    def __init__(self, log_sources):
        self.log_sources = log_sources
        self.es_client = None
        self.splunk_session = None

    async def retrieve(self, api_calls):
        tasks = [self.process_api_call(call) for call in api_calls]
        results = await asyncio.gather(*tasks, return_exceptions=True)

        logs = {}
        for call, result in zip(api_calls, results):
            if isinstance(result, Exception):
                logger.error(f"Error retrieving logs for {call['target_log_source']}: {str(result)}")
            else:
                logs[call['target_log_source']] = result

        return logs

    async def process_api_call(self, api_call):
        source = api_call['target_log_source']
        if source == 'application_logs':
            return await self.get_elasticsearch_logs(api_call)
        elif source == 'security_logs':
            return await self.get_splunk_logs(api_call)
        else:
            raise ValueError(f"Unsupported log source: {source}")

    @async_retry_with_backoff(max_attempts=3, backoff_in_seconds=1)
    async def get_elasticsearch_logs(self, api_call):
        if not self.es_client:
            es_config = next(source for source in self.log_sources if source['name'] == 'application_logs')
            self.es_client = AsyncElasticsearch(
                [es_config['url']],
                http_auth=(es_config['username'], es_config['password']),
                timeout=es_config['timeout']
            )

        query = build_elasticsearch_query(api_call)

        try:
            result = await self.es_client.search(index=api_call['query_parameters'].get('index', '*'), body=query,
                                                 size=1000)
            return [hit['_source'] for hit in result['hits']['hits']]
        except Exception as e:
            logger.error(f"Elasticsearch query failed: {str(e)}")
            raise

    @async_retry_with_backoff(max_attempts=3, backoff_in_seconds=1)
    async def get_splunk_logs(self, api_call):
        if not self.splunk_session:
            splunk_config = next(source for source in self.log_sources if source['name'] == 'security_logs')
            self.splunk_session = aiohttp.ClientSession(headers={'Authorization': f'Bearer {splunk_config["token"]}'})

        search_query = self.build_splunk_query(api_call)

        try:
            async with self.splunk_session.post(f"{splunk_config['url']}/services/search/jobs",
                                                data={'search': search_query}) as response:
                job_id = await response.json()['sid']

            while True:
                async with self.splunk_session.get(f"{splunk_config['url']}/services/search/jobs/{job_id}") as response:
                    if await response.json()['isDone']:
                        break
                await asyncio.sleep(1)

            async with self.splunk_session.get(f"{splunk_config['url']}/services/search/jobs/{job_id}/results",
                                               params={'output_mode': 'json'}) as response:
                results = await response.json()

            return results['results']
        except Exception as e:
            logger.error(f"Splunk query failed: {str(e)}")
            raise

    def build_splunk_query(self, api_call):
        return f"""
        search index=* 
        earliest={api_call['query_parameters'].get('start_time')} 
        latest={api_call['query_parameters'].get('end_time')}
        {api_call['query_parameters'].get('search_terms', '')}
        """

    async def close(self):
        if self.es_client:
            await self.es_client.close()
        if self.splunk_session:
            await self.splunk_session.close()


# Example usage
async def main():
    log_sources = [
        {
            "name": "application_logs",
            "type": "elasticsearch",
            "url": "http://elasticsearch:9200",
            "username": "es_user",
            "password": "es_password",
            "timeout": 30
        },
        {
            "name": "security_logs",
            "type": "splunk",
            "url": "https://splunk.example.com:8089",
            "token": "splunk_token",
            "timeout": 30
        }
    ]

    api_calls = [
        {
            "target_log_source": "application_logs",
            "query_parameters": {
                "start_time": "2023-07-07T00:00:00Z",
                "end_time": "2023-07-07T23:59:59Z",
                "search_terms": "login AND (failure OR suspicious)"
            }
        },
        {
            "target_log_source": "security_logs",
            "query_parameters": {
                "start_time": "2023-07-07T00:00:00Z",
                "end_time": "2023-07-07T23:59:59Z",
                "search_terms": "firewall AND blocked"
            }
        }
    ]

    log_retrieval = LogRetrievalEngine(log_sources)
    logs = await log_retrieval.retrieve(api_calls)
    print(json.dumps(logs, indent=2))
    await log_retrieval.close()


if __name__ == "__main__":
    asyncio.run(main())
