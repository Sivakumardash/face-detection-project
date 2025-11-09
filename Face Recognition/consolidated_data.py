import cv2
import os
import pickle
import numpy as np

data_dir = os.path.join(os.getcwd(),'data')
img_dir = os.path.join(os.getcwd(),'images')


image_data = []
labels = []

for i in os.listdir(img_dir):                    # lists all filenames in the images folder. Example: ['Siva_1.jpg', 'John_1.jpg'].
    image = cv2.imread(os.path.join(img_dir,i))  #reads the image as a NumPy array in BGR format.
    image = cv2.resize(image,(100,100))           #used to resize into 100*100 pixels 
    image = cv2.cvtColor(image,cv2.COLOR_BGR2GRAY)# convert bgr to gray
    image_data.append(image)                      #adds processed image to the list.
    labels.append(str(i).split("_")[0])           # 'Siva_1.jpg'.split("_")[0] → 'Siva'
    
image_data = np.array(image_data)    
labels = np.array(labels) 

print(image_data)
print(labels)

import matplotlib.pyplot as plt
plt.imshow(image_data[60],cmap="gray")
plt.show()


with open(os.path.join(data_dir,"images.p"),'wb') as f:
    pickle.dump(image_data,f)
    
with open(os.path.join(data_dir,"labels.p"),'wb') as f:
    pickle.dump(labels,f)
    
'''
Main Files

Face_Detection code.ipynb → Jupyter notebook (probably for testing face detection).

collect_data.py → Collects images from camera.

consolidated_data.py → Converts collected images into training data (pickle format).

recognize.py → Runs real-time face recognition.

haarcascade_frontalface_default.xml → Pre-trained Haar Cascade model for detecting faces.

'''
