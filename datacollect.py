import cv2

cam = cv2.VideoCapture(0)


cascPath = "haarcascade_frontalface_default.xml"
faceCascade = cv2.CascadeClassifier(cascPath)

face_id = input("\nInput id: ")

print("\n Initial Camera")

count = 0

while (True):
    ret, frame = cam.read()
    frame = cv2.flip(frame, 1)
    
    gray = cv2.cvtColor(frame, cv2.COLOR_BGR2GRAY)
    
    faces = faceCascade.detectMultiScale(
        gray,
        scaleFactor=1.3,
        minNeighbors=5,
        minSize=(30, 30)
    )
    # Draw a rectangle around the faces
    for (x, y, w, h) in faces:
        
        cv2.rectangle(frame, (x, y), (x+w, y+h), (0, 255, 0), 2)
        
        count += 1
        
        cv2.imwrite("dataset/user." + str(face_id) + '.' + str(count) + ".jpeg", gray[y:y+h, x:x+w])

        cv2.imshow("Data Collecting", frame)
    
    k = cv2.waitKey(100) & 0xff
    if k == 27:
        break
    elif count >= 500:
        break
    
print("\nExitting....")

cam.release()
cv2.destroyAllWindows()

