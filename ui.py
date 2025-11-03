#INTERFACE UTILISATEUR v1.2_alpha


import streamlit
import requests
from PIL import Image
import io
import base64
import numpy
import matplotlib.pyplot
import cv2


#URL de l'API FastAPI
#URL_API = "{api_url}"            #VERSION GOOGLE COLAB
URL_API = "http://localhost:8001" #VERSION AZURE
streamlit.set_page_config(page_title="VGG16-Unet & YOLO Interface", layout="wide")
streamlit.title("P🧠C")

#Fonction de conversion base64 --> Image
def base64VersImage(b64_string):
    donneesImage = base64.b64decode(b64_string)
    return Image.open(io.BytesIO(donneesImage))

#Fonction de recolorisation des masques
def reColorisation(imageBrute, cmap="viridis"):
    if(isinstance(imageBrute, Image.Image)):
        imageBrute = numpy.array(imageBrute.convert("L"))
    if(imageBrute.dtype in [numpy.float32, numpy.float64]):
        imageBrute = (imageBrute * 255).astype(numpy.uint8)
    imageBrute = (imageBrute * (255 // 7)).astype(numpy.uint8)
    colorMap = matplotlib.pyplot.get_cmap(cmap)
    imageColorMapee = colorMap(imageBrute)
    imageTraitee = (imageColorMapee[:, :, :3] * 255).astype(numpy.uint8)
    return Image.fromarray(imageTraitee)

#Fonction de fusion entre l'image caméra et le masque prédit
def fusionImageCameraEtMasquePredit(imageCamera, masquePredit, alpha=0.4):
    imageCamera_array = numpy.array(imageCamera)
    masquePredit_array = numpy.array(masquePredit.convert("L"))
    masquePredit_resized = cv2.resize(masquePredit_array.astype(float), (imageCamera_array.shape[1], imageCamera_array.shape[0]), interpolation=cv2.INTER_LINEAR)
    matplotlib.pyplot.figure(figsize=(10, 6))
    matplotlib.pyplot.imshow(imageCamera_array)
    matplotlib.pyplot.imshow(masquePredit_resized, cmap="viridis", alpha=alpha)
    matplotlib.pyplot.axis("off")
    matplotlib.pyplot.tight_layout(pad=0)
    bufferTemp = io.BytesIO()
    matplotlib.pyplot.savefig(bufferTemp, format="png", bbox_inches="tight", pad_inches=0, dpi=100)
    bufferTemp.seek(0)
    imageFusion = Image.open(bufferTemp)
    matplotlib.pyplot.close()  
    return imageFusion

#Définition des choix possibles
choix = streamlit.radio("Que voulez-vous faire ?", ["Lister les fichiers de test", "Prédire sur un numéro", "Uploader et prédire une image"])

if(choix == "Lister les fichiers de test"):
    if(streamlit.button("Afficher la liste des fichiers")):
        try:
            res = requests.post(f"{URL_API}/list", timeout=30)
            if(res.status_code == 200):
                fichiers = res.json()["fichiers"]
                streamlit.success(f"✅ Nombre de fichiers : {len(fichiers)}")
                streamlit.table(fichiers)
            else:
                streamlit.error(f"Erreur API : {res.text}")
        except Exception as e:
            streamlit.error(f"Erreur : {str(e)}")

elif(choix == "Prédire sur un numéro"):
    numeroImage = streamlit.number_input("Numéro de l'image :", min_value=0, step=1)
    streamlit.write("Choix du/des modèle(s) :")
    selectionModeleVGG = streamlit.checkbox("VGG16-Unet (ancien modèle)", value=True)
    selectionModeleYOLO = streamlit.checkbox("YOLOv8 (nouveau modèle)", value=False)
    modeles = []
    if(selectionModeleVGG):
        modeles.append("VGG16-Unet")
    if(selectionModeleYOLO):
        modeles.append("YOLOv8")
    lancementPrediction = streamlit.button("Lancer la prédiction")
    if((lancementPrediction) and (len(modeles) > 0)):
        for modele in modeles:
            if(modele == "VGG16-Unet"):
                endpoint = "/predict"
            else:
                endpoint = "/predict_YOLO"
            streamlit.markdown(f"🔍 Résultat du modèle {modele}")
            try:
                res = requests.post(f"{URL_API}{endpoint}", json={"numero": numeroImage}, timeout=60)
                if(res.status_code == 200):
                    donnees = res.json()
                    imageCamera = base64VersImage(donnees["imageCamera"])
                    masqueReel = reColorisation(base64VersImage(donnees["masqueReel"]))
                    masquePredit = reColorisation(base64VersImage(donnees["masquePredit"]))
                    imageFusion = fusionImageCameraEtMasquePredit(imageCamera, base64VersImage(donnees["masquePredit"]))
                    col1, col2, col3, col4 = streamlit.columns(4)
                    with col1:
                        streamlit.image(imageCamera, caption="Caméra", width=350)
                    with col2:
                        streamlit.image(masqueReel, caption="Masque réel", width=350)
                    with col3:
                        streamlit.image(masquePredit, caption="Masque prédit", width=350)
                    with col4:
                        streamlit.image(imageFusion, caption="Fusion", width=350)
                else:
                    streamlit.error(f"Erreur : {res.text}")
            except Exception as e:
                streamlit.error(f"Erreur : {str(e)}")

elif(choix == "Uploader et prédire une image"):
    fichier = streamlit.file_uploader("Choisir un fichier", type=["png", "jpg", "jpeg"])
    if(fichier is not None):
        image = Image.open(fichier)
        streamlit.write("Choix du/des modèle(s) :")
        selectionModeleVGG = streamlit.checkbox("VGG16-Unet (ancien modèle)", value=True, key="upload_VGG")
        selectionModeleYOLO = streamlit.checkbox("YOLOv8 (nouveau modèle)", value=False, key="upload_YOLO")
        modeles = []
        if(selectionModeleVGG):
            modeles.append("VGG16-Unet")
        if(selectionModeleYOLO):
            modeles.append("YOLOv8")
        lancementPrediction = streamlit.button("Lancer la prédiction")
        if((lancementPrediction) and (len(modeles) > 0)):
            for modele in modeles:
                if(modele == "VGG16-Unet"):
                    endpoint = "/predict_upload"
                else:
                    endpoint = "/predict_upload_YOLO"
                streamlit.markdown(f"🔍 Résultat du modèle {modele}")
                try:
                    fichier.seek(0)
                    fichierOctets = fichier.read()
                    files = {"file": (fichier.name, fichierOctets, fichier.type)}
                    res = requests.post(f"{URL_API}{endpoint}", files=files, timeout=60)
                    if(res.status_code == 200):
                        donnees = res.json()
                        masquePredit = reColorisation(base64VersImage(donnees["masquePredit"]))
                        imageFusion = fusionImageCameraEtMasquePredit(image, base64VersImage(donnees["masquePredit"]))
                        col1, col2, col3 = streamlit.columns(3)
                        with col1:
                            streamlit.image(image, caption="Image uploadée", width=400)
                        with col2:
                            streamlit.image(masquePredit, caption="Masque prédit", width=400)
                        with col3:
                            streamlit.image(imageFusion, caption="Fusion", width=400)
                    else:
                        streamlit.error(f"Erreur : {res.text}")
                except Exception as e:
                    streamlit.error(f"Erreur : {str(e)}")
