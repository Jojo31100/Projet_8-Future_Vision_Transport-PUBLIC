#API v0.1 (LIST only)


#Imports
import os
import numpy
from PIL import Image
from fastapi import FastAPI
from fastapi import FastAPI, UploadFile, File
from fastapi.middleware.cors import CORSMiddleware
from pydantic import BaseModel
import tensorflow
import base64
import io
import tensorflow
from tensorflow.keras.applications import VGG16
from tensorflow.keras.layers import Input, Conv2D, MaxPooling2D, UpSampling2D, concatenate
from tensorflow.keras.models import Model
from tensorflow.keras import backend
from tensorflow.keras.models import load_model


#Chargement du modèle
cheminModele = "./model/best_model_VGG16Unet_sans_DataAugmentation.keras"
#DEBUG repertoireDonneesDeTest = "/content/drive/My Drive/Colab_Notebooks/Project_8/dataset/New_dataset/test/"
repertoireDonneesDeTest = "."
modelCharge = load_model(cheminModele, compile=False)
app = FastAPI(title="VGG16-Unet API")
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
    return {"message": "API VGG16-Unet v0.1 - en ligne"}

#Endpoint : Lister les fichiers de test
@app.post("/list")
def listeFichiers():
    fichiers = sorted([fichier for fichier in os.listdir(repertoireDonneesDeTest) if fichier.endswith("_leftImg8bit.png")])
    return {"nb fichiers": len(fichiers), "fichiers": fichiers}

#Endpoint : Prédire sur un numéro
@app.post("/predict")
def predict(request: PredictionRequest):
    fichiersCamera = sorted([fichier for fichier in os.listdir(repertoireDonneesDeTest) if fichier.endswith("_leftImg8bit.png")])
    fichiersMasques = sorted([fichier for fichier in os.listdir(repertoireDonneesDeTest) if fichier.endswith("_gtFine_labelIds.png")])
    index = request.numero - 1
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

#Endpoint : Uploader et prédire une image
@app.post("/predict_upload")
def predict_upload(file: UploadFile = File(...)):
    #Charger l'image uploadée
    imageUploadee = Image.open(file.file).convert("RGB")
    #Préparation de l'image pour modèle VGG16 (224x224 pixels en entrée)
    imagePrediction = numpy.array(imageUploadee.resize((224,224))).astype("float32")/255
    imagePrediction = numpy.expand_dims(imagePrediction, axis=0)
    prediction = modelCharge.predict(imagePrediction, verbose=0)
    masquePredit_array = numpy.argmax(prediction[0], axis=-1)
    return {"masquePredit": imageVersBase64(resizeHorizontalX2(masquePredit_array))}
