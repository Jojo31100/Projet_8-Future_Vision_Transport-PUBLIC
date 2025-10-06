![CI/CD](https://github.com/Jojo31100/Projet_8-Future_Vision_Transport-PUBLIC/actions/workflows/main_openclassrooms-api.yml/badge.svg)
---
# Projet_8-Future_Vision_Transport-PUBLIC
Projet OpenClassRooms n°8 : Future Vision Transport - Traitez les images pour le système embarqué d’une voiture autonome - PUBLIC
---
## Structure du projet :
- `.github/workflows/main_openclassrooms-api.yml` → Scripts GitHub Actions
- `FutureVisionTransport_Notebook.ipynb` → Notebook Google Colab
- `api.py` → API Prod : VGG16 & Unet v1.0 - FastAPI
- `ui.py` → UI Prod : Interface graphique utilisateur v1.0 - StreamLit
- `azure.yaml` → Fichier de configuration Azure App Services
- `README.md` → Ce fichier
- `requirements.txt` → Définition des packages de pré-requis pour faire tourner l'API et la UI
- `runtime.txt` → Définition de la version de Python nécessaire sur l'EndPoint Azure
- `startup.sh` → Script de démarrage de l'API
- ---
L'UI sert à communiquer avec l'API (modèle "VGG16+Unet"), pré-entraînée et modifiée pour catégoriser les pixels d'une image parmi 7 classes principales (`void`, `flat`, `construction`, `object`, `nature`, `sky`, `human`, `vehicle`)

Attendus :

- Méthode <u>**GET**</u> :

  - Retour : `{"message": "API VGG16-Unet v1.0 - en ligne"}`


- Méthode <u>**POST**</u> : *https://<URL_ACCES_API>/*`list`

  - Retour : liste des images disponibles pour test


- Méthode <u>**POST**</u> : *https://<URL_ACCES_API>/*`predict`

  - Retour : masque prédit pour une image existante (voir /list)  
  - Exemple JSON : `{"numero": 3}`


- Méthode <u>**POST**</u> : *https://<URL_ACCES_API>/*`predict_upload`

  - Retour : masque prédit pour une image uploadée  
  - Exemple : `{"file": <image>}`
