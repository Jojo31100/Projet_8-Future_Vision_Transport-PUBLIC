import threading
import sys
import uvicorn
import os
from api import app

#Lance l'API FastAPI sur un port interne (non exposé Azure)
def run_api():
    uvicorn.run(app, host="0.0.0.0", port=8001)

#Lance Streamlit (UI) sur le port externe Azure
def run_streamlit():
    import streamlit.web.cli as stcli
    port = os.getenv("PORT", "8501")
    sys.argv = [
        "streamlit", "run", "ui.py",
        "--server.port", port,
        "--server.headless", "true",
        "--server.address", "0.0.0.0",
    ]
    stcli.main()

if __name__ == "__main__":
    threading.Thread(target=run_api, daemon=True).start()
    run_streamlit()
