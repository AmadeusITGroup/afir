"""
AFIR entrypoint: runs identically on a laptop and as a Databricks App.

  Local:            python app.py
  Databricks App:   declared as the `command` in app.yaml

`main.py` uses flat imports (``from anomaly_detection import ...``) that resolve
with ``src/`` on the path, so we add it before importing. The HTTP port comes from
``DATABRICKS_APP_PORT`` when the platform injects it, else from config.
"""

import asyncio
import os
import sys
from pathlib import Path

# Make src/ importable so main.py's flat imports resolve from any cwd.
SRC = Path(__file__).resolve().parent / "src"
sys.path.insert(0, str(SRC))

from main import main  # noqa: E402

if __name__ == "__main__":
    port = os.getenv("DATABRICKS_APP_PORT")
    if port:
        print(f"Starting AFIR on Databricks App port {port}", flush=True)
    asyncio.run(main())
