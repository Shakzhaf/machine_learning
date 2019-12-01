import numpy as np
import cv2

# multiple cascades: https://github.com/Itseez/opencv/tree/master/data/haarcascades

#https://github.com/Itseez/opencv/blob/master/data/haarcascades/haarcascade_frontalface_default.xml
fist_cascade = cv2.CascadeClassifier('fist.xml')
palm_cascade1 = cv2.CascadeClassifier('closed_frontal_palm.xml')
#https://github.com/Itseez/opencv/blob/master/data/haarcascades/haarcascade_eye.xml
palm_cascade = cv2.CascadeClassifier('palm.xml')

cap = cv2.VideoCapture(cv2.CAP_DSHOW)

while 1:
    ret, img = cap.read()
    gray = cv2.cvtColor(img, cv2.COLOR_BGR2GRAY)
    fists = fist_cascade.detectMultiScale(gray, 1.3, 5)
    #print(fists)
    for (x,y,w,h) in fists:
        cv2.rectangle(img,(x,y),(x+w,y+h),(255,0,0),2)
        #roi_gray = gray[y:y+h, x:x+w]
        #roi_color = img[y:y+h, x:x+w]
        
    palms = palm_cascade.detectMultiScale(gray, 1.3, 5)
    palms1= palm_cascade1.detectMultiScale(gray,1.3, 5)
    #print(palms)
    for (ex,ey,ew,eh) in palms:
        cv2.rectangle(img,(ex,ey),(ex+ew,ey+eh),(0,255,0),2)

    for (ex1,ey1,ew1,eh1) in palms:
        cv2.rectangle(img,(ex1,ey1),(ex1+ew1,ey1+eh1),(0,255,0),2)

    cv2.imshow('img',img)
    k = cv2.waitKey(30) & 0xff
    if k == 27:
        break

cap.release()
cv2.destroyAllWindows()