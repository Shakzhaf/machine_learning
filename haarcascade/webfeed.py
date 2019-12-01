import numpy as np 
import cv2

cap = cv2.VideoCapture(cv2.CAP_DSHOW)
ret = True
path=np.zeros((300,300))
path[True]=False
while (ret):
	ret,image = cap.read()
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
	try:
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
	except:
		pass	
	#cv2.drawContours(image, cnt, 0, (0,255,0), 3)
	cv2.imshow('th',th)
	cv2.imshow('image',image)
	cv2.imshow("path",path)
	if cv2.waitKey(25) & 0xFF == ord('q'):
	  cv2.destroyAllWindows()
	  cap.release()
	  break
	if cv2.waitKey(25) & 0xFF == ord('a'):
	  path[True]=False