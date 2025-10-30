#!/bin/bash
set -e

#Répertoire racine de l’application
APP_PATH="/home/site/wwwroot"
cd "$APP_PATH"

#Trouver et activer un environnement virtuel si présent
VENV_PATH=$(find /home -type d -name "antenv" | head -n 1)
if [ -d "$VENV_PATH" ]; then
    echo "Activation du virtualenv à $VENV_PATH"
    source "$VENV_PATH/bin/activate"
else
    echo "⚠️  Aucun virtualenv trouvé, exécution dans l'environnement global."
fi

#Port unique exposé par Azure
APP_PORT=${PORT:-8501}
echo "Démarrage de l’application FastAPI + Streamlit sur le port $APP_PORT"

#Lancer le script principal (UI + API dans un même processus)
exec python3 ui_launcher.py
