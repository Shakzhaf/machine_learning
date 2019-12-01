from __future__ import print_function
import keras
from keras.datasets import mnist
from keras.layers import Dense, Flatten
from keras.layers import Conv2D, MaxPooling2D
from keras.models import Sequential
from keras.models import model_from_yaml
import numpy as np
from PIL import Image
import matplotlib.pyplot as plt
from matplotlib import style
style.use('fivethirtyeight')
#get_ipython().run_line_magic('matplotlib', 'inline')
import glob
import os
from collections import Counter
import pandas as pd
import time 
import cv2

yaml_file = open('mnist_model.yaml', 'r')
loaded_model_yaml = yaml_file.read()
yaml_file.close()
model = model_from_yaml(loaded_model_yaml)
# load weights into new model
model.load_weights("mnist_model.h5")
print("Loaded model from disk")
model.compile(loss=keras.losses.categorical_crossentropy,
              optimizer=keras.optimizers.Adam(),
              metrics=['accuracy'])

def predict(image,show=True):
    im=np.array(image)
    pr = model.predict_classes(im.reshape((1,28,28,1)))
    answer=pr.tolist()[0]
    if show:
        plt.imshow(im)
        plt.show()
    #print(answer)
    return answer

i=0
while(i<20):
	feed_image=np.ones((28,28))
	print(predict(feed_image,show=False))
	i+=1