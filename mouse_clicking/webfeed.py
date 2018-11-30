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

font = cv2.FONT_HERSHEY_SIMPLEX
bottomLeftCornerOfText = (10,50)
fontScale = 1
fontColor = (255,255,255)
lineType  = cv2.LINE_AA

cap = cv2.VideoCapture(cv2.CAP_DSHOW)
ret = True
path=np.zeros((300,300))
path[True]=False
while (ret):
	ret,image = cap.read()
	read_path = cv2.imread('path.jpg',0)
	image = cv2.flip(image, 1)
	image = cv2.resize(image,(300,300))
	gray = cv2.cvtColor(image, cv2.COLOR_BGR2GRAY)
	blur =cv2.GaussianBlur(gray,(11,11),0)
	th=cv2.adaptiveThreshold(blur,255,cv2.ADAPTIVE_THRESH_GAUSSIAN_C,\
            cv2.THRESH_BINARY_INV,11,2)
	#ret,th=cv2.threshold(blur,125,255,cv2.THRESH_BINARY_INV)
	dilated=cv2.dilate(th,np.ones((3,3),np.uint8),iterations=3)
	something, contours, hierarchy =cv2.findContours(dilated, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)
	cnt = sorted(contours, key=cv2.contourArea, reverse=True)
	
	blur_path =cv2.GaussianBlur(read_path,(11,11),0)
	ret,th_path=cv2.threshold(blur_path,125,255,cv2.THRESH_BINARY_INV)
	something1, contours1, hierarchy1 =cv2.findContours(th_path, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)
	cnt_path = sorted(contours1, key=cv2.contourArea, reverse=True)

	try:
		digit = cv2.boundingRect(cnt_path[0])
		x_d,y_d,w_d,h_d=digit
		cv2.rectangle(path,(x_d,y_d),(x_d+w_d,y_d+h_d),(0,255,0),2)
		
		feed_image = read_path[y_d:y_d+h_d, x_d:x_d+w_d]

		feed_image = cv2.resize(feed_image,(28,28))
		#feed_image = cv2.cvtColor(feed_image, cv2.COLOR_BGR2GRAY)
		
		cv2.imshow('feed_image',feed_image)
		#print('im running')
		answer=predict(feed_image,show=False)
		#cv2.putText(image,answer, bottomLeftCornerOfText, font, fontScale,fontColor,lineType)
		print(answer)
	except:
		pass


	try:
		if(cv2.isContourConvex(contours)):
			print('convex')

		rect = cv2.boundingRect(cnt[0])
		x,y,w,h = rect
		cv2.rectangle(image,(x,y),(x+w,y+h),(0,255,0),2)
		#cv2.rectangle(path,(mid_x-2,mid_y-2),(mid_x+2,mid_y+2),(0,255,0),2)
		mid_x=int(x+(w/2))
		mid_y=int(y+(h/3))
		cv2.rectangle(image,(mid_x-1,mid_y-1),(mid_x+1,mid_y+1),(255,255,255),2)
		cv2.rectangle(path,(mid_x-1,mid_y-1),(mid_x+1,mid_y+1),(255,255,255),2)
		#path[mid_x][mid_y]=True
		#frame[path]=[0,0,0]
		#print(mid_x,",",mid_y,",",path[mid_x][mid_y])
		
		
			#print("not convex")
	except:
		pass	
	#cv2.drawContours(image, cnt, 0, (0,255,0), 3)
	cv2.imwrite('path.jpg',path)
	cv2.imshow('th',th)
	cv2.imshow('image',image)
	cv2.imshow("path",path)
	if cv2.waitKey(25) & 0xFF == ord('q'):
	  cv2.destroyAllWindows()
	  cap.release()
	  break
	if cv2.waitKey(25) & 0xFF == ord('a'):
	  path[True]=False