import cv2
import os

recognizer = cv2.face.LBPHFaceRecognizer.create()
recognizer.read('trainer/trainer.yml')


cascPath = "haarcascade_frontalface_default.xml"
faceCascade = cv2.CascadeClassifier(cascPath)

font = cv2.FONT_HERSHEY_COMPLEX

id = 0

names = ["Biden", "Mark", "Lam", "Trump", "Giang", "Musk", "Vy", "An", "Thanh", "Hieu"]



video_capture = cv2.VideoCapture(0)

while True:
    ret, frame = video_capture.read()
    
    frame = cv2.flip(frame, 1)
    
    if ret != True:
        break
    
    gray = cv2.cvtColor(frame, cv2.COLOR_BGR2GRAY)
    
    faces = faceCascade.detectMultiScale(gray,
                                         scaleFactor=1.2,
                                         minNeighbors=5,
                                         minSize=(30, 30))
    
    
    for (x,y,w,h) in faces:
        
        cv2.rectangle(frame, (x, y), (x + w, y + h), (0, 255, 0), 0)
        
        id, confidence = recognizer.predict(gray[y:y+h, x:x+w])
        
        if confidence < 100:
            id = names[id]
            confidence = " {0}%".format(round(100 - confidence))
        else:
            id = "unknown!"
            confidence = " {0}%".format(round(100 - confidence))
            
            
        cv2.putText(frame, str(id), (x + 5, y - 5), font, 1, (255, 255, 255), 2)
        cv2.putText(frame, str(confidence), (x + 5, y + h - 5), font, 1, (255, 255, 0), 1)
        
    cv2.imshow("Facial Recognizer", frame)
    
    k = cv2.waitKey(1) & 0xff
    if k == 27:
        break


print("\n[INFO] Quiting...")

video_capture.release()

cv2.destroyAllWindows()