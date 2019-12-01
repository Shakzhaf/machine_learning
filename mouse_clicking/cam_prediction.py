
# coding: utf-8

# In[ ]:


#importing necessary libraries
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




# In[ ]:


# load YAML and create model
yaml_file = open('fist_palm_cnn_model.yaml', 'r')
loaded_model_yaml = yaml_file.read()
yaml_file.close()
model = model_from_yaml(loaded_model_yaml)
# load weights into new model
model.load_weights("fist_palm_cnn_model.h5")
print("Loaded model from disk")
model.compile(loss=keras.losses.categorical_crossentropy,
              optimizer=keras.optimizers.Adam(),
              metrics=['accuracy'])


# In[ ]:


def predict(image,show=True):
    im=np.array(image)
    pr = model.predict_classes(im.reshape((1,100,100,1)))
    answer=pr.tolist()[0]
    if show:
        plt.imshow(im)
        plt.show()
    #print(answer)
    return answer


# In[ ]:


prediction={0:'palm',1:'fist'}
cap = cv2.VideoCapture(cv2.CAP_DSHOW)
ret = True
while (ret):
    ret,image = cap.read()
  # read image as grey scale
    image_np=cv2.resize(image,(100,100))
    gray = cv2.cvtColor(image_np, cv2.COLOR_BGR2GRAY)
    #ret,thresh_img = cv2.threshold(gray,127,255,cv2.THRESH_BINARY)
    thresh_img=cv2.adaptiveThreshold(gray,255,cv2.ADAPTIVE_THRESH_GAUSSIAN_C,\
            cv2.THRESH_BINARY,11,2)
    #thresh_img2=cv2.adaptiveThreshold(gray,255,cv2.ADAPTIVE_THRESH_MEAN_C,\
    #        cv2.THRESH_BINARY,11,2)
    #gray=image_np.convert('LA')
    #status = cv2.imwrite('train/palm/palm'+str(i)+'.jpg',cv2.resize(gray,(150,150)))
    #print(image_np.shape,',',gray.shape)
    #i+=1
    cv2.imshow('image',image)
    cv2.imshow('binary_image',thresh_img)
    #cv2.imshow('binary_image2',thresh_img2)
    i=predict(thresh_img,False)
    print(prediction[i])
    if cv2.waitKey(25) & 0xFF == ord('q'):
        cv2.destroyAllWindows()
        cap.release()
        break

