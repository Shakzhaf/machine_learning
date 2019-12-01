import cv2
import numpy as np
import os
cap = cv2.VideoCapture(0)
ret = True
i=0
while (i<500):
	ret,image_np = cap.read()
  # read image as grey scale
	gray = cv2.cvtColor(image_np, cv2.COLOR_BGR2GRAY)
 
# save image
	image=cv2.resize(gray,(300,300))
	status = cv2.imwrite('train1/fist/palm'+str(i)+'.jpg',cv2.resize(gray,(150,150)))
	i+=1
	cv2.imshow('image',image)
	if cv2.waitKey(25) & 0xFF == ord('q'):
		cv2.destroyAllWindows()
		cap.release()
		break
