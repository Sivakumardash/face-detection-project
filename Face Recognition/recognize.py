
import urllib
import cv2
import numpy as np
from keras.models import load_model

classifier = cv2.CascadeClassifier('haarcascade_frontalface_default.xml')

model = load_model("final_model.h5")

URL = 'http://172.26.38.97:8080/shot.jpg'

def get_pred_label(pred):
    labels = ["akash","chandru","rakesh","shiva","siddhant"]
    return labels[pred]

def preprocess(img):
    img = cv2.cvtColor(img,cv2.COLOR_BGR2GRAY)
    img = cv2.resize(img,(100,100))
    img = cv2.equalizeHist(img)
    img = img.reshape(1,100,100,1)
    img = img/255
    return img

ret = True
while ret:
    
    img_url = urllib.request.urlopen(URL)
    image = np.array(bytearray(img_url.read()),np.uint8)
    frame = cv2.imdecode(image,-1)
    
    faces = classifier.detectMultiScale(frame,1.5,5)
      
    for x,y,w,h in faces:
        face = frame[y:y+h,x:x+w]
        cv2.rectangle(frame,(x,y),(x+w,y+h),(255,0,0),5)
        cv2.putText(frame,get_pred_label(np.argmax(model.predict(preprocess(face)))),
                    (200,200),cv2.FONT_HERSHEY_COMPLEX,1,
                    (255,0,0),2)
        
    cv2.imshow("capture",frame)
    if cv2.waitKey(1)==ord('q'):
        break

cv2.destroyAllWindows()


'''
 Face Recognition/
 └── Face Recognition/
      ├── collect_data.py        ← used to capture faces & build dataset
      ├── consolidated_data.py   ← used to combine dataset & prepare pickle files
      ├── recognize.py           ← MAIN FILE (runs face recognition live)
      ├── haarcascade_frontalface_default.xml
      ├── data/
      │     ├── images.p
      │     ├── labels.p
      ├── images/ (your training face images)
      
      
      Main Files

Face_Detection code.ipynb → Jupyter notebook (probably for testing face detection).

collect_data.py → Collects images from camera.

consolidated_data.py → Converts collected images into training data (pickle format).

recognize.py → Runs real-time face recognition.

haarcascade_frontalface_default.xml → Pre-trained Haar Cascade model for detecting faces.
'''