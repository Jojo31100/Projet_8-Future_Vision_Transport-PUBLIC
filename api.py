#API v1.1_alpha


#Imports
import os
import numpy
from PIL import Image
from fastapi import FastAPI, UploadFile, File
from fastapi.middleware.cors import CORSMiddleware
from pydantic import BaseModel
import base64
import io
import tensorflow
from tensorflow.keras.applications import VGG16
from tensorflow.keras.layers import Input, Conv2D, MaxPooling2D, UpSampling2D, concatenate
from tensorflow.keras.models import Model
from tensorflow.keras import backend
from tensorflow.keras.models import load_model
from fastapi.responses import JSONResponse
from ultralytics import YOLO
import cv2


#Chargement du modèle
cheminModele = os.path.join(os.path.dirname(__file__), "model", "best_model_VGG16Unet_sans_DataAugmentation.keras") 
cheminModele_YOLO = os.path.join(os.path.dirname(__file__), "model", "best_model_YOLOv8-l-seg_finetuned.pt")
repertoireDonneesDeTest = os.path.join(os.path.dirname(__file__), "testPictures")
modelCharge = load_model(cheminModele, compile=False)
modelCharge_YOLO = YOLO(cheminModele_YOLO)
app = FastAPI(title="VGG16-Unet & YOLOv8 API")
app.add_middleware(CORSMiddleware, allow_origins=["*"], allow_credentials=True, allow_methods=["*"], allow_headers=["*"])

#Classe pour la prédiction
class PredictionRequest(BaseModel):
    numero: int

#Fonction de conversion Image --> base64
def imageVersBase64(imageArray):
    image = Image.fromarray(imageArray.astype(numpy.uint8))
    buffered = io.BytesIO()
    image.save(buffered, format="PNG")
    return base64.b64encode(buffered.getvalue()).decode("utf-8")

#Resize pour affichage l'image au bon ratio
def resizeHorizontalX2(image_array):
    image = Image.fromarray(image_array.astype(numpy.uint8))
    image_resized = image.resize((image_array.shape[1]*2, image_array.shape[0]), resample=Image.NEAREST)
    return numpy.array(image_resized)

#Endpoint racine
@app.get("/")
async def root():
    return {"message": "API VGG16-Unet & YOLOv8 v1.1_alpha"}

#Endpoint VGG16+Unet & YOLO : Lister les fichiers de test
@app.post("/list")
def listeFichiers():
    fichiers = sorted([fichier for fichier in os.listdir(repertoireDonneesDeTest) if fichier.endswith("_leftImg8bit.png")])
    return {"Nombre de fichiers": len(fichiers), "fichiers": fichiers}

#Endpoint VGG16+Unet : Prédire sur un numéro
@app.post("/predict")
def predict(request: PredictionRequest):
    fichiersCamera = sorted([fichier for fichier in os.listdir(repertoireDonneesDeTest) if fichier.endswith("_leftImg8bit.png")])
    fichiersMasques = sorted([fichier for fichier in os.listdir(repertoireDonneesDeTest) if fichier.endswith("_gtFine_labelIds.png")])
    index = request.numero
    if((index < 0) or (index >= len(fichiersCamera))):
        return {"error": "Numéro d'image invalide !"}
    fichierCameraSelectionne = os.path.join(repertoireDonneesDeTest, fichiersCamera[index])
    fichierMasqueSelectionne = os.path.join(repertoireDonneesDeTest, fichiersMasques[index])
    imageCamera_array = numpy.array(Image.open(fichierCameraSelectionne))
    imageMasque_array = numpy.array(Image.open(fichierMasqueSelectionne))
    #Préparation de l'image pour modèle VGG16 (224x224 pixels en entrée)
    imagePrediction = numpy.array(Image.fromarray(imageCamera_array).resize((224,224))).astype("float32")/255
    imagePrediction = numpy.expand_dims(imagePrediction, axis=0)
    prediction = modelCharge.predict(imagePrediction, verbose=0)
    masquePredit_array = numpy.argmax(prediction[0], axis=-1)
    return {"imageCamera": imageVersBase64(imageCamera_array), "masqueReel": imageVersBase64(imageMasque_array), "masquePredit": imageVersBase64(resizeHorizontalX2(masquePredit_array)), "numero_image": request.numero}

#Endpoint YOLO : Prédire sur un numéro
@app.post("/predict_YOLO")
def predict_YOLO(request: PredictionRequest):
    fichiersCamera = sorted([fichier for fichier in os.listdir(repertoireDonneesDeTest) if fichier.endswith("_leftImg8bit.png")])
    fichiersMasques = sorted([fichier for fichier in os.listdir(repertoireDonneesDeTest) if fichier.endswith("_gtFine_labelIds.png")])
    index = request.numero
    if((index < 0) or (index >= len(fichiersCamera))):
        return {"error": "Numéro d'image invalide !"}
    fichierCameraSelectionne = os.path.join(repertoireDonneesDeTest, fichiersCamera[index])
    fichierMasqueSelectionne = os.path.join(repertoireDonneesDeTest, fichiersMasques[index])
    imageCamera_array = numpy.array(Image.open(fichierCameraSelectionne))
    imageMasque_array = numpy.array(Image.open(fichierMasqueSelectionne))
    #Inférence YOLO
    predictions = modelCharge_YOLO.predict(fichierCameraSelectionne, verbose=False)
    #Récupération taille originale de l'image
    height, width = imageMasque_array.shape[:2]
    masquePreditFusionne_8 = numpy.zeros((height, width), dtype=int)
    #YOLO travaille en segmentation par instance, et non pas segmentation par classe...
    if((predictions[0].masks is not None) and (len(predictions[0].masks.data) > 0)):
        for indexInstance, masqueInstance in enumerate(predictions[0].masks.data):
            #Classe directement prédite par le modèle (déjà dans le système 0-7)
            classe_8 = int(predictions[0].boxes.cls[indexInstance].cpu().numpy())
            #Masque binaire 2D
            masqueInstance_array = masqueInstance.cpu().numpy()
            if(masqueInstance_array.ndim == 3):
                masqueInstance_array = masqueInstance_array[0]  #Car parfois (1,Height,Width) au lieu de (Height,Width)
            masqueInstance_bool = masqueInstance_array.astype(bool)
            #Redimensionnement du masque d’instance pour correspondre à la taille du "masque réel"
            masqueInstance_resized = cv2.resize(masqueInstance_bool.astype(numpy.uint8), (width, height), interpolation=cv2.INTER_NEAREST).astype(bool)
            #Fusion de toutes les instances d’une même classe
            masquePreditFusionne_8[masqueInstance_resized] = classe_8
    return {"imageCamera": imageVersBase64(imageCamera_array), "masqueReel": imageVersBase64(imageMasque_array), "masquePredit": imageVersBase64(masquePreditFusionne_8), "numero_image": request.numero}

#Endpoint VGG16+Unet : Uploader et prédire une image
@app.post("/predict_upload")
async def predict_upload(file: UploadFile = File(...)):
    try:
        #Lire tout le contenu en mode "octets"
        fichierOctets = await file.read()
        if(len(fichierOctets) == 0):
            return {"error": "Fichier vide ou non lu correctement"}
        imageUploadee = Image.open(io.BytesIO(fichierOctets)).convert("RGB")
        #Préparation pour le modèle
        imagePrediction = numpy.array(imageUploadee.resize((224,224))).astype("float32")/255
        imagePrediction = numpy.expand_dims(imagePrediction, axis=0)
        prediction = modelCharge.predict(imagePrediction, verbose=0)
        masquePredit_array = numpy.argmax(prediction[0], axis=-1)
        return {"masquePredit": imageVersBase64(resizeHorizontalX2(masquePredit_array))}
    except Exception as excpt:
        return {"error": str(excpt)}

@app.post("/predict_upload_YOLO")
async def predict_upload_YOLO(file: UploadFile = File(...)):
    try:
        fichierOctets = await file.read()
        if(len(fichierOctets) == 0):
            return {"error": "Fichier vide ou non lu correctement"}
        imageUploadee = Image.open(io.BytesIO(fichierOctets)).convert("RGB")
        imageUploadee_array = numpy.array(imageUploadee)
        height, width = imageUploadee_array.shape[:2]
        #Inférence YOLO
        predictions = modelCharge_YOLO.predict(numpy.array(imageUploadee), verbose=False)
        masquePreditFusionne_8 = numpy.zeros((height, width), dtype=int)
        #YOLO travaille en segmentation par instance, et non pas segmentation par classe...
        if((predictions[0].masks is not None) and (len(predictions[0].masks.data) > 0)):
            for indexInstance, masqueInstance in enumerate(predictions[0].masks.data):
                #Classe directement prédite par le modèle (déjà dans le système 0-7)
                classe_8 = int(predictions[0].boxes.cls[indexInstance].cpu().numpy())
                #Masque binaire 2D
                masqueInstance_array = masqueInstance.cpu().numpy()
                if(masqueInstance_array.ndim == 3):
                    masqueInstance_array = masqueInstance_array[0]  #Car parfois (1,Height,Width) au lieu de (Height,Width)
                masqueInstance_bool = masqueInstance_array.astype(bool)
                #Redimensionnement du masque d’instance pour correspondre à la taille du "masque réel"
                masqueInstance_resized = cv2.resize(masqueInstance_bool.astype(numpy.uint8), (width, height), interpolation=cv2.INTER_NEAREST).astype(bool)
                #Fusion de toutes les instances d’une même classe
                masquePreditFusionne_8[masqueInstance_resized] = classe_8
        return {"masquePredit": imageVersBase64(masquePreditFusionne_8)}
    except Exception as excpt:
        return {"error": str(excpt)}
