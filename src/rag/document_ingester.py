import asyncio
import json
import logging
from pathlib import Path
from typing import Any, Dict, List

import docx
import pandas as pd
import PyPDF2

logger = logging.getLogger(__name__)


class DocumentIngester:
    """Ingests various document formats for RAG system"""

    SUPPORTED_FORMATS = {".pdf", ".docx", ".doc", ".txt", ".csv", ".json"}

    async def ingest_directory(self, directory_path: str) -> List[Dict[str, Any]]:
        """Ingest all supported documents from a directory"""
        documents = []
        path = Path(directory_path)

        if not path.exists():
            logger.error(f"Directory {directory_path} does not exist")
            return documents

        for file_path in path.rglob("*"):
            if (
                file_path.is_file()
                and file_path.suffix.lower() in self.SUPPORTED_FORMATS
            ):
                doc = await self.ingest_file(str(file_path))
                if doc:
                    documents.append(doc)

        logger.info(f"Ingested {len(documents)} documents from {directory_path}")
        return documents

    async def ingest_file(self, file_path: str) -> Dict[str, Any]:
        """Ingest a single document file"""
        try:
            file_path = Path(file_path)
            suffix = file_path.suffix.lower()

            if suffix == ".pdf":
                content = await self._read_pdf(file_path)
            elif suffix in [".docx", ".doc"]:
                content = await self._read_docx(file_path)
            elif suffix == ".txt":
                content = await self._read_text(file_path)
            elif suffix == ".csv":
                content = await self._read_csv(file_path)
            elif suffix == ".json":
                content = await self._read_json(file_path)
            else:
                logger.warning(f"Unsupported file format: {suffix}")
                return None

            return {
                "title": file_path.stem,
                "content": content,
                "type": f"document_{suffix[1:]}",
                "metadata": {
                    "file_path": str(file_path),
                    "file_size": file_path.stat().st_size,
                    "modified_time": file_path.stat().st_mtime,
                },
            }

        except Exception as e:
            logger.error(f"Error ingesting file {file_path}: {str(e)}")
            return None

    async def _read_pdf(self, file_path: Path) -> str:
        """Read PDF file content"""
        content = []

        def read_pdf_sync():
            with open(file_path, "rb") as file:
                pdf_reader = PyPDF2.PdfReader(file)
                for page in pdf_reader.pages:
                    content.append(page.extract_text())
            return "\n".join(content)

        return await asyncio.to_thread(read_pdf_sync)

    async def _read_docx(self, file_path: Path) -> str:
        """Read DOCX file content"""

        def read_docx_sync():
            doc = docx.Document(file_path)
            return "\n".join([para.text for para in doc.paragraphs])

        return await asyncio.to_thread(read_docx_sync)

    async def _read_text(self, file_path: Path) -> str:
        """Read text file content"""

        def read_text_sync():
            with open(file_path, "r", encoding="utf-8") as file:
                return file.read()

        return await asyncio.to_thread(read_text_sync)

    async def _read_csv(self, file_path: Path) -> str:
        """Read CSV file content as structured text"""

        def read_csv_sync():
            df = pd.read_csv(file_path)
            content = f"CSV Data from {file_path.name}:\n"
            content += f"Columns: {', '.join(df.columns)}\n"
            content += f"Shape: {df.shape[0]} rows, {df.shape[1]} columns\n\n"
            # Include sample data
            content += "Sample data:\n"
            content += df.head(10).to_string()
            return content

        return await asyncio.to_thread(read_csv_sync)

    async def _read_json(self, file_path: Path) -> str:
        """Read JSON file content"""

        def read_json_sync():
            with open(file_path, "r", encoding="utf-8") as file:
                data = json.load(file)
                return json.dumps(data, indent=2)

        return await asyncio.to_thread(read_json_sync)
