import threading
import sys
import uvicorn
from api import app

#Fonction pour lancer Streamlit dans un thread séparé
def run_streamlit():
    import streamlit.web.cli as stcli
    sys.argv = [
        "streamlit", "run", "ui.py",
        "--server.port", str(PORT),
        "--server.headless", "true",
        "--server.address", "0.0.0.0",
    ]
    stcli.main()

if __name__ == "__main__":
    import os
    PORT = int(os.getenv("PORT", "8501"))

    #Démarrer Streamlit en arrière-plan
    threading.Thread(target=run_streamlit, daemon=True).start()

    #Démarrer FastAPI sur le même port
    uvicorn.run(app, host="0.0.0.0", port=PORT)
