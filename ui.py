#INTERFACE UTILISATEUR (DASHBOARD) v1.3_alpha


import streamlit
import requests
from PIL import Image
import io
import base64
import pandas
import matplotlib.pyplot
import cv2


#URL de l'API FastAPI
#URL_API = "{api_url}"            #VERSION GOOGLE COLAB
URL_API = "http://localhost:8001" #VERSION AZURE


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


#Mise en page & CSS pour réduire les paddings
streamlit.set_page_config(page_title="VGG16-Unet & YOLO Interface", layout="wide")
streamlit.markdown("""<style>.block-container{padding-top:1rem; padding-bottom:0rem; padding-left:1rem; padding-right:1rem;}h1,h2,h3{margin-top:0rem; margin-bottom:0rem; line-height:1.5;}.stMarkdown{margin-bottom:0.3rem;}div[data-testid="stHorizontalBlock"]{gap:0.5rem;}hr{margin-top:0.3rem; margin-bottom:0.5rem;}.stRadio > div{gap:0.3rem;}</style>""", unsafe_allow_html=True)

#Titre principal
streamlit.markdown("<h1 style='text-align:center; margin-bottom:0.2rem;'>P🧠C - DASHBOARD</h1>", unsafe_allow_html=True)

#1ère section : analyse exploratoire
streamlit.markdown("<h3 style='text-align:left; margin-top:0rem; margin-bottom:0.3rem;'>Analyse exploratoire</h3>", unsafe_allow_html=True)
streamlit.markdown("<hr style='border: 3px solid blue;'>", unsafe_allow_html=True)

#Liste des macro-Classes
colonnesDesMacroClasses = [
    "VOID",
    "FLAT",
    "CONSTRUCTION",
    "OBJECT",
    "NATURE",
    "SKY",
    "HUMAN",
    "VEHICLE",
]

#Valeurs de réparition dans les macro-Classes
valeurs = {
    "Train": [10, 39, 22, 2, 15, 3, 1, 7],
    "Val":   [10, 39, 23, 2, 14, 3, 1, 8],
    "Test":  [11, 38, 20, 2, 17, 4, 1, 7]
}

labelsDataset = ["Train", "Val", "Test"]
pourcentageDuDataset = [60, 10, 30]

#Palette de couleurs Matplotlib par défaut (la même que celle utilisée pour le barplot)
paletteCouleurs = matplotlib.pyplot.rcParams["axes.prop_cycle"].by_key()["color"]
troisPremieresCouleurs = paletteCouleurs[:3]

#Création des colonnes Streamlit (30%, 50% et 20%)
col1, col2, _ = streamlit.columns([3, 5, 2])

#Camembert
with col1:
    fig_camembert, ax_camembert = matplotlib.pyplot.subplots(figsize=(6,6), dpi=80)
    wedges, texts, autotexts = ax_camembert.pie(pourcentageDuDataset, labels=labelsDataset, colors=troisPremieresCouleurs, autopct="%1.0f%%", pctdistance=0.5, startangle=0)
    ax_camembert.set_title("Répartition des données Train/Val/Test", fontsize=10)
    for autotext in autotexts:
        autotext.set_color("black")
        autotext.set_fontsize(8)
    buffer_camembert = io.BytesIO()
    fig_camembert.savefig(buffer_camembert, format="png", bbox_inches="tight", dpi=80)
    buffer_camembert.seek(0)
    streamlit.image(buffer_camembert, use_container_width=False)
    matplotlib.pyplot.close(fig_camembert)

#Histogramme
with col2:
    #Transformation en DataFrame (pour se faciliter la vie)
    dataframe = pandas.DataFrame(valeurs, index=colonnesDesMacroClasses)
    fig_histogramme, ax_histogramme = matplotlib.pyplot.subplots(figsize=(8,5), dpi=80)
    dataframe.plot(kind="bar", edgecolor="black", ax=ax_histogramme)
    ax_histogramme.set_title("Répartition des pixels par macro-Classes", fontsize=10)
    ax_histogramme.set_ylabel("Pourcentage (%)", fontsize=8)
    ax_histogramme.set_ylim(0, 50)
    ax_histogramme.set_xlabel("Macro-Classes", fontsize=8)
    ax_histogramme.set_xticklabels(colonnesDesMacroClasses, rotation=75, fontsize=7)
    ax_histogramme.legend(title="Dataset", fontsize=8)
    for pourcentage in ax_histogramme.patches:
        height = pourcentage.get_height()
        if(height > 0):
            ax_histogramme.annotate(f"\n---{height:.0f}%", (pourcentage.get_x() + pourcentage.get_width() / 2, height), ha="center", va="bottom", fontsize=7, rotation=75)
    matplotlib.pyplot.tight_layout()
    buffer_histogramme = io.BytesIO()
    fig_histogramme.savefig(buffer_histogramme, format="png", bbox_inches="tight", dpi=80)
    buffer_histogramme.seek(0)
    streamlit.image(buffer_histogramme, use_container_width=False)
    matplotlib.pyplot.close(fig_histogramme)

#2ème section : test des modèles
streamlit.markdown("<h3 style='text-align:left; margin-top:0rem; margin-bottom:0.3rem;'>Test des modèles</h3>", unsafe_allow_html=True)
streamlit.markdown("<hr style='border: 3px solid blue;'>", unsafe_allow_html=True)

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
