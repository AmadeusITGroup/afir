import asyncio
import logging

from elasticsearch import AsyncElasticsearch
from sshtunnel import SSHTunnelForwarder

from src.utils.error_handling import async_retry_with_backoff

logger = logging.getLogger(__name__)


def build_elasticsearch_query(api_call):
    query = {
        "bool": {
            "must": [
                {"term": {"officeId": "AGAMO2301"}},
                {"term": {"user.userId": "AGALEZA"}}
            ],
            "filter": [
                {"range": {"date": {"gte": "2024-09-19", "lt": "2024-09-20"}}}
            ]
        }
    }
    return query


class LogRetrievalEngine:
    def __init__(self, config):
        self.config = config
        self.es_client = None

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

    async def retrieve_with_tunnel(self, api_calls):
        with SSHTunnelForwarder(
                self.config['tunnel']['url'],
                ssh_username="userHolder".replace('userHolder', self.config['tunnel']['user']),
                ssh_password="pwdHolder".replace('pwdHolder', self.config['tunnel']['password']),
                remote_bind_address=(self.config['tunnel']['remote_bind_url'], self.config['tunnel']['remote_bind_port']),
                local_bind_address=(self.config['tunnel']['local_bind_url'], self.config['tunnel']['local_bind_port'])
        ) as server:
            server.start()
            tasks = [self.process_api_call(call) for call in api_calls]
            results = await asyncio.gather(*tasks, return_exceptions=True)

            logs = {}
            for call, result in zip(api_calls, results):
                if isinstance(result, Exception):
                    logger.error(f"Error retrieving logs for {call['target_log_source']}: {str(result)}")
                else:
                    logs[call['target_log_source']] = result

            server.stop()
            return logs

    async def process_api_call(self, api_call):
        source_type = ""
        source = api_call['target_log_source']
        if source in self.config["names_list"]:
            for option in self.config["sources"]:
                if option['name'] == source:
                    source_type = option["type"]
        else:
            raise ValueError(f"Unsupported log source: {source}")

        match source_type:
            case "elasticsearch":
                return await self.get_elasticsearch_logs(api_call)
            case _:
                raise ValueError(f"Unsupported source type: {source_type}")

    @async_retry_with_backoff(max_attempts=3, backoff_in_seconds=1)
    async def get_elasticsearch_logs(self, api_call):
        if not self.es_client:
            es_config = next(source for source in self.config["sources"] if source['type'] == 'elasticsearch')
            self.es_client = AsyncElasticsearch(
                es_config["url"],
                http_auth=(es_config["username"], es_config["password"]),
                timeout=es_config['timeout']
            )

        query = build_elasticsearch_query(api_call)

        try:
            result = await self.es_client.search(index='session.azure.ic.2024.09', query=query,
                                                 size=1000)
            return [hit['_source'] for hit in result['hits']['hits']]
        except Exception as e:
            logger.error(f"Elasticsearch query failed: {str(e)}")
            raise


# Example usage
async def main():
    return

if __name__ == "__main__":
    asyncio.run(main())
