#!/bin/bash
set -e #Stoppe en cas d'erreur

APP_PATH="/home/site/wwwroot"
cd "$APP_PATH"

# Chercher le venv
VENV_PATH=$(find /home -type d -name "antenv" | head -n 1)
if [ -d "$VENV_PATH" ]; then
    echo "Activation du virtualenv à $VENV_PATH"
    source "$VENV_PATH/bin/activate"
    export PATH="$VENV_PATH/bin:$PATH"
    echo "Virtualenv activé, Python: $(which python)"
else
    echo "⚠️  Aucun virtualenv trouvé, exécution dans l'environnement système."
fi

# Ports
API_PORT=8001
SL_PORT=${PORT:-8501}

# Lancer FastAPI en arrière-plan
echo "Démarrage de FastAPI sur le port $API_PORT..."
nohup uvicorn api:app --host 0.0.0.0 --port $API_PORT > /home/logs/fastapi.log 2>&1 &

# Log de diagnostic
echo "Flux de logs : $(ls -l /home/logs 2>/dev/null || true)"
echo "PATH actuel = $PATH"
which streamlit || echo "⚠️ Streamlit introuvable dans le PATH"

# Lancer Streamlit au premier plan
echo "Démarrage de Streamlit sur le port $SL_PORT..."
exec streamlit run ui.py --server.port "$SL_PORT" --server.headless true
