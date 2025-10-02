#INTERFACE UTILISATEUR v0.1 (LIST only)


#Imports
import streamlit
import requests
from PIL import Image
import io
import base64


#URL de l'API FastAPI Azure
URL_API = "http://localhost:8000" #On ne peut pas passer par l'URL publique (le port 8000 sera fermé, il faut donc rester sur la VM et le 127.0.0.1/localhost, port 8000 pour accéder à l'API)
streamlit.set_page_config(page_title="VGG16-Unet Interface", layout="wide")
streamlit.title("Future Vision Transport")

#Fonction de conversion base64 --> Image
def base64VersImage(b64_string):
    donneesImage = base64.b64decode(b64_string)
    return Image.open(io.BytesIO(donneesImage))

#Définition des choix possibles
choix = streamlit.radio("Que voulez-vous faire ?", ["Lister les fichiers de test ?", "Prédire sur un numéro ?", "Uploader et prédire une image ?"])

#Lister les fichiers de test
if(choix == "Lister les fichiers de test ?"):
    if(streamlit.button("Afficher la liste des fichiers")):
        res = requests.post(f"{URL_API}/list")
        if(res.status_code == 200):
            fichiers = res.json()["fichiers"]
            streamlit.write(f"Nombre de fichiers : {len(fichiers)}")
            streamlit.table(fichiers)
        else:
            streamlit.error(f"Erreur API : {res.text}")

#Prédire sur un numéro
elif(choix == "Prédire sur un numéro ?"):
    numeroImage = streamlit.number_input("Numéro de l'image à prédire :", min_value=1, step=1)
    if(streamlit.button("Lancer la prédiction")):
        res = requests.post(f"{URL_API}/predict", json={"numero": numeroImage})
        if(res.status_code == 200):
            data = res.json()
            streamlit.image(base64VersImage(data["imageCamera"]), caption="Image de la caméra")
            streamlit.image(base64VersImage(data["masqueReel"]), caption="Masque réel")
            streamlit.image(base64VersImage(data["masquePredit"]), caption="Masque prédit")
        else:
            streamlit.error(f"Erreur API : {res.text}")

#Uploader et prédire une image
elif(choix == "Uploader et prédire une image ?"):
    fichier = streamlit.file_uploader("Choisir un fichier", type=["png", "jpg", "jpeg"])
    if(fichier is not None):
        image = Image.open(fichier)
        if(streamlit.button("Lancer la prédiction sur l'image uploadée")):
            #Envoi du fichier à l'API
            res = requests.post(f"{URL_API}/predict_upload", files={"file": fichier})
            if(res.status_code == 200):
                data = res.json()
                #Création de 2 colonnes pour afficher l'image et son masque prédit, côte à côte
                colonne1, colonne2 = streamlit.columns(2)
                with colonne1:
                    streamlit.image(image, caption="Image uploadée")
                with colonne2:
                    streamlit.image(base64VersImage(data["masquePredit"]), caption="Masque prédit")
            else:
                streamlit.error(f"Erreur API : {res.text}")
