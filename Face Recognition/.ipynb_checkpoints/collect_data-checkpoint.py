'''
import cv2
import urllib
import numpy as np

classifier = cv2.CascadeClassifier(r"C:/Users/Lenovo/Desktop/Face Recognition/Face Recognition/haarcascade_frontalface_default.xml")

url = "http://192.0.0.4:8080/shot.jpg"


data = []

while len(data) < 100:
    
    
    image_from_url = urllib.request.urlopen(url)
    frame = np.array(bytearray(image_from_url.read()),np.uint8)
    frame = cv2.imdecode(frame,-1)
    
    face_points = classifier.detectMultiScale(frame,1.3,5)                            
    
    if len(face_points)>0:
        for x,y,w,h in face_points:
            face_frame = frame[y:y+h+1,x:x+w+1]
            cv2.imshow("Only face",face_frame)
            if len(data)<=100:
                print(len(data)+1,"/100")
                data.append(face_frame)
                break
    cv2.putText(frame, str(len(data)),(100,100),cv2.FONT_HERSHEY_SIMPLEX,5,(0,0,255))
    cv2.imshow("frame",frame)
    if cv2.waitKey(30) == ord("q"):
        break
cv2.destroyAllWindows()
        
if len(data)== 100:
    name = input("Enter Face holder name : ")
    for i in range(100):
        cv2.imwrite("images/"+name+"_"+str(i)+".jpg",data[i])
    print("Done")
else:
    print("need more data")
        
    
'''

import cv2
import numpy as np
import os

# Load Haar cascade classifier
classifier = cv2.CascadeClassifier(r"C:/Users/Lenovo/Desktop/Face Recognition/Face Recognition/haarcascade_frontalface_default.xml")

# Initialize laptop webcam
cap = cv2.VideoCapture(0)  # 0 = default webcam

data = []

# Create folder if it doesn't exist
if not os.path.exists("images"):
    os.makedirs("images")

while len(data) < 100:
    ret, frame = cap.read()  # Capture frame from webcam
    if not ret:
        continue  # Skip if frame not read properly

    face_points = classifier.detectMultiScale(frame, 1.3, 5)

    if len(face_points) > 0:
        for x, y, w, h in face_points:
            face_frame = frame[y:y+h+1, x:x+w+1]
            cv2.imshow("Only face", face_frame)
            if len(data) < 100:
                print(len(data)+1, "/100")
                data.append(face_frame)
                break  # Only take one face per frame

    cv2.putText(frame, str(len(data)), (100, 100), cv2.FONT_HERSHEY_SIMPLEX, 5, (0, 0, 255))
    cv2.imshow("frame", frame)

    if cv2.waitKey(30) & 0xFF == ord("q"):
        break

# Release webcam and close windows
cap.release()
cv2.destroyAllWindows()

if len(data) == 100:
    name = input("Enter Face holder name : ")
    for i in range(100):
        cv2.imwrite("images/" + name + "_" + str(i) + ".jpg", data[i])
    print("Done")
else:
    print("need more data")




'''
Main Files

Face_Detection code.ipynb → Jupyter notebook (probably for testing face detection).

collect_data.py → Collects images from camera.

consolidated_data.py → Converts collected images into training data (pickle format).

recognize.py → Runs real-time face recognition.

haarcascade_frontalface_default.xml → Pre-trained Haar Cascade model for detecting faces.

'''