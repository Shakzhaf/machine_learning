import numpy as np 
import cv2

cap = cv2.VideoCapture(0)
ret = True
while (ret):
	ret,tfImage = cap.read()
	mean_img=tfImage.mean()
	print(mean_img)
	mask=tfImage < mean_img
	print(mask)
	tfImage[True]=255
	tfImage[mask]=0
	cv2.imshow('image',cv2.resize(tfImage,(800,600)))
	if cv2.waitKey(25) & 0xFF == ord('q'):
	  cv2.destroyAllWindows()
	  cap.release()
	  break