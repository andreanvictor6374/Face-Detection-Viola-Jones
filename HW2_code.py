import numpy as np
import cv2

face_cascade = cv2.CascadeClassifier('haarcascade_frontalface_default.xml')
eye_cascade = cv2.CascadeClassifier('haarcascade_eye.xml')
smile_cascade=cv2.CascadeClassifier('haarcascade_smile.xml')

#filename = 'friends'
#filename = 'GOT'
#filename = 'lena'
filename = 'victor'
face_eye_smile=[True,True,True]

img = cv2.imread(filename+'.jpg')
gray = cv2.cvtColor(img, cv2.COLOR_BGR2GRAY)

if face_eye_smile[0]:
    
    faces = face_cascade.detectMultiScale(gray, 1.3, 5) ###
    for (x,y,w,h) in faces:
        img = cv2.rectangle(img,(x,y),(x+w,y+h),(255,0,0),2)
        roi_gray = gray[y:y+h, x:x+w]
        roi_color = img[y:y+h, x:x+w]
        if face_eye_smile[1]:
            eyes = eye_cascade.detectMultiScale(roi_gray,1.1,22)###
            for (ex,ey,ew,eh) in eyes:
                cv2.rectangle(roi_color,(ex,ey),(ex+ew,ey+eh),(0,255,0),2)
        if face_eye_smile[2]:
            smiles=smile_cascade.detectMultiScale(roi_gray,1.9,22) ###
            for (ex,ey,ew,eh) in smiles:
                cv2.rectangle (roi_color,(ex,ey),(ex+ew,ey+eh),(0,0,255),2) 

list_=['face','_eye','_smile']
out_name='_'
for i,e in enumerate(face_eye_smile):
    if e==True:
        out_name=out_name+list_[i]
        
cv2.imwrite(filename+out_name+'.jpg', img) 

cv2.imshow('img',img)
#cv2.waitKey(0)
cv2.destroyAllWindows()





