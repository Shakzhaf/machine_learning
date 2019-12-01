import numpy as np 
import cv2

cap = cv2.VideoCapture(0)
ret = True
ret,frame1 = cap.read()
frame1=cv2.resize(frame1,(300,300))
ret,frame2 = cap.read()
frame2=cv2.resize(frame2,(300,300))
path=np.zeros((300,300))
path[True]=False

while ret:
	ret,frame = cap.read()
	frame=cv2.resize(frame,(300,300))
	frame=cv2.flip(frame, 1)
	print(frame.shape)
      #VideoFileOutput.write(frame)
	d=cv2.absdiff(frame1,frame2)
	grey=cv2.cvtColor(d,cv2.COLOR_BGR2GRAY)
	blur =cv2.GaussianBlur(grey,(5,5),0)
	ret,th=cv2.threshold(blur,20,255,cv2.THRESH_BINARY)
	dilated=cv2.dilate(th,np.ones((3,3),np.uint8),iterations=3)
	img,c,h=cv2.findContours(dilated,cv2.RETR_TREE,cv2.CHAIN_APPROX_SIMPLE)
	cnt = sorted(c, key=cv2.contourArea, reverse=True)
	try:
		rect = cv2.boundingRect(cnt[0])
		x,y,w,h = rect
		cv2.rectangle(frame1,(x,y),(x+w,y+h),(0,255,0),2)
		#cv2.rectangle(path,(mid_x-2,mid_y-2),(mid_x+2,mid_y+2),(0,255,0),2)
		mid_x=int(x+(w/2))
		mid_y=int(y+(h/2))
		cv2.rectangle(path,(mid_x-1,mid_y-1),(mid_x+1,mid_y+1),(255,255,255),2)
		#path[mid_x][mid_y]=True
		#frame[path]=[0,0,0]
		#print(mid_x,",",mid_y,",",path[mid_x][mid_y])
	except:
		pass	
	#cv2.drawContours(frame1,cnt,0,(0,255,0),2)
	cv2.imshow("inter",frame1)
	cv2.imshow("path",path)
	#cv2.imshow("difference",dilated)
	#cv2.imshow("th",th)
	frame1 = frame2
	ret,frame2= cap.read()
	frame2=cv2.resize(frame2,(300,300))
	frame2=cv2.flip(frame2, 1)
	if cv2.waitKey(25) & 0xFF == ord('q'):
	  cv2.destroyAllWindows()
	  cap.release()
	  break
	if cv2.waitKey(25) & 0xFF == ord('a'):
	  path[True]=False
