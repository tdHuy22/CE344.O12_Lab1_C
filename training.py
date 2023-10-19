import cv2
import os
import numpy as np
from PIL import Image


path = 'dataset'

recognizer = cv2.face.LBPHFaceRecognizer.create()
cascPath = "haarcascade_frontalface_default.xml"

detector = cv2.CascadeClassifier(cascPath)

def getImagesAndLabels(path):
    imagePaths = [os.path.join(path, f) for f in os.listdir(path)]

    faceSamples=[]
    faceIDs=[]
    
    for imagePath in imagePaths:
        PIL_img = Image.open(imagePath).convert('L')
        img_numpy = np.array(PIL_img, 'uint8')
        
        id = int(os.path.split(imagePath)[-1].split(".")[1])
        
        faces = detector.detectMultiScale(img_numpy)
        for(x,y,w,h) in faces:
            faceSamples.append(img_numpy[y:y+h, x:x+w])
            faceIDs.append(np.array(id))
    
    return faceSamples, faceIDs
            
print("\nTraining the model....")

faces, ids = getImagesAndLabels(path)
recognizer.train(faces, np.array(ids))

recognizer.write("trainer/trainer.yml")

print("INFO {0} faces have been trained. OUT".format(len(np.unique(ids))))