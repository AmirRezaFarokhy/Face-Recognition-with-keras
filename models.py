import cv2
import numpy as np
import os
import random
import matplotlib.pyplot as plt

import tensorflow as tf
from tensorflow.keras.models import Sequential
from tensorflow.keras.layers import Dense, Conv2D, MaxPooling2D, Dropout, Flatten
from tensorflow.keras.layers import Activation, BatchNormalization

NAME = ["Amir Reza Farokhy, You have access", "Not Recognize, access denied!!!"]

PATH = ["pos", "neg"]
SIZE = (85, 85)

BATCH_SIZE = 16
EPOCHS = 8

def ReadProcessingData(path):
    dataset = []
    for target, p in enumerate(path):
        path_files = os.listdir(p)
        for files in path_files:
            path_files_joins = os.path.join(p, files)
            img = cv2.imread(path_files_joins)
            img = img/255.0 # scalig data
            dataset.append([img, target])

    # shuffle data
    random.shuffle(dataset)

    X = []
    y = []
    for features, target in dataset:
        X.append(features)
        y.append(target)
        
    return np.array(X), np.array(y)


x_train, y_train = ReadProcessingData(PATH)

model = Sequential()

model.add(Conv2D(32, (3,3), input_shape=x_train.shape[1:]))
model.add(Activation("relu"))
model.add(BatchNormalization())
model.add(Dropout(0.2))
model.add(MaxPooling2D(pool_size=(2,2)))

model.add(Conv2D(32,(3,3)))
model.add(Activation("relu"))
model.add(BatchNormalization())
model.add(Dropout(0.1))
model.add(MaxPooling2D(pool_size=(2,2)))

model.add(Flatten())

model.add(Dense(16))
model.add(Activation("relu"))

# output layer
model.add(Dense(1))
model.add(Activation("sigmoid"))

model.summary()

model.compile(loss='binary_crossentropy',
              optimizer='adam',
              metrics=['accuracy'])

model.summary()


hist = model.fit(x_train, y_train, 
                epochs=EPOCHS,
                batch_size=BATCH_SIZE,
                validation_split=0.1)

# plot loss function
plt.plot(hist.history["loss"], label="train_loss")
plt.plot(hist.history["val_loss"], label="val_loss")
plt.legend(loc="upper right")
plt.show()

# plot accuracy
plt.plot(hist.history["accuracy"], label="train_acc")
plt.plot(hist.history["val_accuracy"], label="val_acc")
plt.legend(loc="upper right")
plt.show()

model.save('my_model.h5')
