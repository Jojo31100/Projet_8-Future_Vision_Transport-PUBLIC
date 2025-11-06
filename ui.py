#INTERFACE UTILISATEUR (DASHBOARD) v2.0


import streamlit
import requests
from PIL import Image
import io
import base64
import matplotlib.pyplot
import cv2
import pandas
import numpy


#URL de l'API FastAPI
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

#Fonction de remappage des classes 34 Cityscapes, vers nos 7 macro-Classes
def mappingDesClassesCityscapeVers8MacroClasses(masque_array):
    mappingDesClassesCityscape = {
        -1: 0,  0: 0,  1: 0,  2: 0,  3: 0,  4: 0,  5: 0,  6: 0, #Void
         7: 1,  8: 1,  9: 1, 10: 1,                             #Flat
        11: 2, 12: 2, 13: 2, 14: 2, 15: 2, 16: 2,               #Construction
        17: 3, 18: 3, 19: 3, 20: 3,                             #Object
        21: 4, 22: 4,                                           #Nature
        23: 5,                                                  #Sky
        24: 6, 25: 6,                                           #Human
        26: 7, 27: 7, 28: 7, 29: 7, 30: 7, 31: 7, 32: 7, 33: 7  #Vehicle
    }
    masque_mappe = numpy.zeros_like(masque_array, dtype=numpy.uint8)
    for id_original, classe_8 in mappingDesClassesCityscape.items():
        masque_mappe[masque_array == id_original] = classe_8
    return masque_mappe

#Fonction de calcul de l'"Intersection Over Union coefficient" (IoU) par catégorie/macro-classe
def coefIoUParClasse(y_true, y_pred, nbClasses=8, smooth=1):
    iouParClasse = {}
    iouClassesPresentes = []
    for classe in range(nbClasses):
        intersection = numpy.sum((y_true == classe) & (y_pred == classe))
        union = numpy.sum((y_true == classe) | (y_pred == classe))
        if(union > 0):
            iou = (intersection + smooth) / (union + smooth)
            iouParClasse[classe] = iou
            iouClassesPresentes.append(iou)
        else:
            iouParClasse[classe] = None #Si la classe est "absente"
    if(len(iouClassesPresentes) > 0):
        iouMoyen = numpy.mean(iouClassesPresentes)
    else:
        iouMoyen = 0
    return iouParClasse, iouMoyen

#Fonction interne pour créer le graphique IoU
def graphiqueIoU(iouParClasse, cmap="viridis"):
    #Préparation des données
    macroClasses = ["VOID", "FLAT", "CONSTRUCTION", "OBJECT", "NATURE", "SKY", "HUMAN", "VEHICLE"]
    iouValeurs = [iouParClasse.get(classe, 0) for classe in range(8)]
    mIoU = numpy.mean([val for val in iouValeurs if val is not None])
    #Création du graphique
    histogramme, ax_histogramme = matplotlib.pyplot.subplots(figsize=(6,4), dpi=100)
    #Palette viridis
    colorMap = matplotlib.pyplot.get_cmap(cmap)
    couleurs = [colorMap(i/7) for i in range(8)]
    #Histogramme
    bars = ax_histogramme.bar(macroClasses, iouValeurs, color=couleurs, edgecolor="black")
    ax_histogramme.set_ylim(0,1)
    ax_histogramme.set_ylabel("IoU", fontsize=8)
    ax_histogramme.set_xlabel("Macro-Classes", fontsize=8)
    ax_histogramme.set_title(f"IoU global : {mIoU*100:.1f}%", fontsize=10)
    #Rotation des labels des abscisses
    ax_histogramme.set_xticklabels(macroClasses, rotation=75, fontsize=8)
    #Valeurs au-dessus des barres
    for bar, val in zip(bars, iouValeurs):
        ax_histogramme.text(bar.get_x() + bar.get_width()/2, val + 0.02, f"{val:.2f}", ha="center", fontsize=7)
    matplotlib.pyplot.tight_layout()
    #Conversion en Image PIL
    bufferTemp = io.BytesIO()
    matplotlib.pyplot.savefig(bufferTemp, format="png", bbox_inches="tight", dpi=100)
    bufferTemp.seek(0)
    imageGraphique = Image.open(bufferTemp)
    matplotlib.pyplot.close(histogramme)
    return imageGraphique

#Fonction factice pour forcer le re-rendu de la page
def placeholder_callback():
    pass


#Mise en page
streamlit.set_page_config(page_title="VGG16-Unet & YOLO Interface", layout="wide")


#1ère section : Analyse Exploratoire
with streamlit.expander("📊 Analyse exploratoire", expanded=True):
    streamlit.markdown("<hr style='border: 3px solid blue;'>", unsafe_allow_html=True)
    #Liste des macro-Classes
    colonnesDesMacroClasses = ["VOID", "FLAT", "CONSTRUCTION", "OBJECT", "NATURE", "SKY", "HUMAN", "VEHICLE"]
    #Les valeurs ci-dessous, certes ici "en dur", sont consultables dans le notebook du projet n°8 "Future Vision Transport"
    #Valeurs de réparition entre les macro-Classes
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

    #Camembert de réparition du dataset
    with col1:
        fig_camembert, ax_camembert = matplotlib.pyplot.subplots(figsize=(6,6), dpi=100)
        wedges, texts, autotexts = ax_camembert.pie(pourcentageDuDataset, labels=labelsDataset, colors=troisPremieresCouleurs, autopct="%1.0f%%", pctdistance=0.5, startangle=0)
        ax_camembert.set_title("Répartition des données Train/Val/Test", fontsize=10)
        for autotext in autotexts:
            autotext.set_color("black")
            autotext.set_fontsize(8)
        buffer_camembert = io.BytesIO()
        fig_camembert.savefig(buffer_camembert, format="png", bbox_inches="tight", dpi=100)
        buffer_camembert.seek(0)
        streamlit.image(buffer_camembert, use_container_width=False)
        matplotlib.pyplot.close(fig_camembert)

    #Histogramme de réparition entre les macro-Classes
    with col2:
        #Transformation en DataFrame (pour se faciliter la vie)
        dataframe = pandas.DataFrame(valeurs, index=colonnesDesMacroClasses)
        fig_histogramme, ax_histogramme = matplotlib.pyplot.subplots(figsize=(7,5), dpi=100)
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
        fig_histogramme.savefig(buffer_histogramme, format="png", bbox_inches="tight", dpi=100)
        buffer_histogramme.seek(0)
        streamlit.image(buffer_histogramme, use_container_width=False)
        matplotlib.pyplot.close(fig_histogramme)

#2ème section : Test des Modèles
with streamlit.expander("🧪 Test des modèles", expanded=True):
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
        selectionModeleYOLO = streamlit.checkbox("YOLOv8 (nouveau modèle)", value=True)
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
                        #Récupération des images brutes
                        imageCamera = base64VersImage(donnees["imageCamera"])
                        masqueReel_brut = base64VersImage(donnees["masqueReel"])
                        masquePredit_brut = base64VersImage(donnees["masquePredit"])
                        #Conversion en arrays numpy
                        masqueReel_array = numpy.array(masqueReel_brut)
                        masquePredit_array = numpy.array(masquePredit_brut)
                        if(masquePredit_array.shape != masqueReel_array.shape):
                            masquePredit_array = cv2.resize(masquePredit_array, (masqueReel_array.shape[1], masqueReel_array.shape[0]), interpolation=cv2.INTER_NEAREST)
                        masqueReel_array_mappe = mappingDesClassesCityscapeVers8MacroClasses(masqueReel_array)
                        #Calcul des IoU (maintenant qu'ils sont bien redimensionnés et mappés)
                        iouParClasse, miou = coefIoUParClasse(masqueReel_array_mappe, masquePredit_array)
                        #Création du graphique IoU
                        imageIoU = graphiqueIoU(iouParClasse, cmap="viridis")
                        #Création des images colorisées pour affichage
                        masqueReel_colore = reColorisation(masqueReel_array_mappe)
                        masquePredit_colore = reColorisation(masquePredit_array)
                        imageFusion = fusionImageCameraEtMasquePredit(imageCamera, Image.fromarray(masquePredit_array))
                        #Affichage
                        col1, col2, col3, col4, col5 = streamlit.columns(5)
                        with col1:
                            streamlit.image(imageCamera, caption="Caméra", width=350)
                        with col2:
                            streamlit.image(masqueReel_colore, caption="Masque réel", width=350)
                        with col3:
                            streamlit.image(masquePredit_colore, caption="Masque prédit", width=350)
                        with col4:
                            streamlit.image(imageFusion, caption="Fusion", width=350)
                        with col5:
                            streamlit.image(imageIoU, width=350)
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
            selectionModeleYOLO = streamlit.checkbox("YOLOv8 (nouveau modèle)", value=True, key="upload_YOLO")
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

#3ème section : Accessibilité
if("tailleTexte" not in streamlit.session_state):
    streamlit.session_state.tailleTexte = 100
with streamlit.expander("♿ Accessibilité", expanded=False):
    streamlit.markdown("<hr style='border: 3px solid blue;'>", unsafe_allow_html=True)
    col_acc1, _ = streamlit.columns([1,1])
    with col_acc1:
        streamlit.slider("Taille du texte (%)", min_value=50, max_value=150, step=10, help="Ajustez ici la taille de tous les textes de la page !", key="tailleTexte", on_change=placeholder_callback)
#Bloc CSS Dynamique
streamlit.markdown(f"""<style>h1, h2, h3 {{margin-top: 0rem !important; margin-bottom: 0rem !important; line-height: 1.2 !important;}}.stApp, .stApp * {{font-size: {streamlit.session_state.tailleTexte / 100}rem !important;}}h1{{font-size: 2.4em !important;}}h2{{font-size: 2.0em !important;}}h3{{font-size: 1.6em !important;}}</style>""", unsafe_allow_html=True)
