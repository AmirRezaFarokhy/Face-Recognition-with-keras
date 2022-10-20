import cv2
import os
import numpy as np
import matplotlib.pyplot as plt

from tensorflow.keras.models import load_model

from ImageProcessing import FaceDetections

model = load_model('my_model.h5')

test = 'test'
target_pred = []
for p in os.listdir(test):
    path_test = os.path.join(test, p)
    img = cv2.imread(path_test)
    img = cv2.resize(img, SIZE).astype(np.float32)
    img = img/255.0
    img = img.reshape(1, img.shape[0], img.shape[1], img.shape[2])
    target_pred.append(NAME[np.argmax(model.predict(img))])
    
    
iamge_examples = cv2.imread(f"{test}/son.jpg")
face_pred = FaceDetections(iamge_examples, SIZE)
iamge_examples = face_pred.Predictions(iamge_examples, target_pred[1])
iamge_examples = face_pred.ConvertToRGB(iamge_examples)
plt.figure(figsize=(18, 18))
plt.imshow(iamge_examples)
plt.show()
