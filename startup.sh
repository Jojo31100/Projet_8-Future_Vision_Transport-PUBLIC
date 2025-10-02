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

#Définition des Ports
API_PORT=8001
SL_PORT=${PORT:-8501}


#Etape 1 : lancer FastAPI en arrière-plan sur le port 8001
echo "Demarrage de FastAPI en arriere-plan sur le port $API_PORT..."
nohup uvicorn api:app --host 0.0.0.0 --port $API_PORT &

#Etape 2 : lancer Streamlit au premier plan sur le port $PORT (exposé par Azure)
echo "Demarrage de Streamlit au premier plan sur le port $ST_PORT..."
streamlit run ui.py --server.port $SL_PORT --server.headless true
