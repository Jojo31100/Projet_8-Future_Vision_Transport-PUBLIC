#Etape 1 : Image de base
FROM python:3.12-slim

#Etape 2 : Variables d'environnement
ENV DEBIAN_FRONTEND=noninteractive
ENV PYTHONDONTWRITEBYTECODE=1
ENV PYTHONUNBUFFERED=1
WORKDIR /app

#Etape 3 : Copier et installer les dépendances
COPY requirements.txt .
RUN apt-get update && apt-get install -y \
    build-essential libgl1 curl git && \
    pip install --upgrade pip && \
    pip install --no-cache-dir -r requirements.txt && \
    apt-get clean && rm -rf /var/lib/apt/lists/*

#Etape 4 : Copier le code
COPY . .

#Etape 5 : Exposer le port pour Streamlit
EXPOSE 8501

#Etape 6 : Lancer FastAPI et Streamlit
CMD bash -c "nohup uvicorn api:app --host 0.0.0.0 --port 8001 & streamlit run ui.py --server.port 8501 --server.headless true"
