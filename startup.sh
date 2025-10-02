#!/bin/bash

#Définir la variable d'environnement pour le chemin de l'application
APP_PATH="/home/site/wwwroot"

#Chercher et activer le Virtual Environment
VENV_PATH=$(find /home -type d -name "antenv")

if [ -n "$VENV_PATH" ]; then
    source "$VENV_PATH/bin/activate"
    echo "Virtual environment active : $VENV_PATH"
else
    echo "Pas de virtual environment trouve. Utilisation de l'environnement systeme."
fi

#Etap 1 : lancer FastAPI en arrière-plan sur le port 8000
echo "Demarrage de FastAPI en arriere-plan..."
nohup uvicorn api:app --host 0.0.0.0 --port 8000 &

#Etape 2 : lancer Streamlit au premier plan sur le port 80 ou celui défini par Azure
#Azure App Service expose le port via la variable $PORT
ST_PORT=${PORT:-8501}
echo "Demarrage de Streamlit au premier plan sur le port $ST_PORT..."
streamlit run ui.py --server.port $ST_PORT --server.headless true
