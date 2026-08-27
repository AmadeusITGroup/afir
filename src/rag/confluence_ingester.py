import asyncio
import logging
from typing import Any, Dict, List

from atlassian import Confluence
from bs4 import BeautifulSoup

logger = logging.getLogger(__name__)


class ConfluenceIngester:
    """Ingests content from Confluence for RAG system"""

    def __init__(self, url: str, username: str, password: str):
        self.confluence = Confluence(url=url, username=username, password=password)

    async def ingest_space(
        self, space_key: str, limit: int = 1000
    ) -> List[Dict[str, Any]]:
        """Ingest all pages from a Confluence space"""
        try:
            pages = await asyncio.to_thread(
                self.confluence.get_all_pages_from_space,
                space_key,
                start=0,
                limit=limit,
                expand="body.storage,version",
            )

            documents = []
            for page in pages:
                doc = await self._process_page(page, space_key)
                if doc:
                    documents.append(doc)

            logger.info(
                f"Ingested {len(documents)} pages from Confluence space {space_key}"
            )
            return documents

        except Exception as e:
            logger.error(f"Error ingesting Confluence space {space_key}: {str(e)}")
            raise

    async def ingest_pages_by_label(
        self, label: str, limit: int = 100
    ) -> List[Dict[str, Any]]:
        """Ingest pages with a specific label"""
        try:
            cql = f'label = "{label}"'
            results = await asyncio.to_thread(
                self.confluence.cql, cql, limit=limit, expand="body.storage,version"
            )

            documents = []
            for page in results.get("results", []):
                doc = await self._process_page(page, label)
                if doc:
                    documents.append(doc)

            logger.info(f"Ingested {len(documents)} pages with label {label}")
            return documents

        except Exception as e:
            logger.error(f"Error ingesting pages with label {label}: {str(e)}")
            raise

    async def _process_page(self, page: Dict[str, Any], context: str) -> Dict[str, Any]:
        """Process a single Confluence page"""
        try:
            page_id = page["id"]
            title = page["title"]

            html_content = page.get("body", {}).get("storage", {}).get("value", "")

            text_content = self._clean_html(html_content)

            version = page.get("version", {}).get("number", 1)
            last_updated = page.get("version", {}).get("when", "")

            page_url = f"{self.confluence.url}/pages/viewpage.action?pageId={page_id}"

            return {
                "title": title,
                "content": text_content,
                "type": "confluence_page",
                "metadata": {
                    "page_id": page_id,
                    "url": page_url,
                    "space_or_label": context,
                    "version": version,
                    "last_updated": last_updated,
                },
            }

        except Exception as e:
            logger.error(
                f"Error processing page {page.get('title', 'Unknown')}: {str(e)}"
            )
            return None

    def _clean_html(self, html: str) -> str:
        """Clean HTML content and extract text"""
        soup = BeautifulSoup(html, "html.parser")

        for script in soup(["script", "style"]):
            script.decompose()

        text = soup.get_text()

        lines = (line.strip() for line in text.splitlines())
        chunks = (phrase.strip() for line in lines for phrase in line.split("  "))
        text = " ".join(chunk for chunk in chunks if chunk)

        return text
