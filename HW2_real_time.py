import cv2
import imageio

#loading cascades
face_cascade=cv2.CascadeClassifier('haarcascade_frontalface_default.xml')
eye_cascade=cv2.CascadeClassifier('haarcascade_eye.xml')
smile_cascade=cv2.CascadeClassifier('haarcascade_smile.xml')

def detect(gray,frame): #x,y =upperleft corner,w=width of rectangle,h=height of rectangle
    faces = face_cascade.detectMultiScale(gray, 1.3, 5)
    for (x,y,w,h) in faces:
        cv2.rectangle (frame,(x,y),(x+w,y+h),(255,0,0),2) #draw a rectangle
        roi_gray=gray[y:y+h,x:x+w]
        roi_color=frame[y:y+h,x:x+w]
        eyes=eye_cascade.detectMultiScale(roi_gray,1.1,22)
        for (ex,ey,ew,eh) in eyes:
            cv2.rectangle (roi_color,(ex,ey),(ex+ew,ey+eh),(0,255,0),2) #(0,255,0):color of the rectangle
        smiles=smile_cascade.detectMultiScale(roi_gray,1.9,22) ###
        for (ex,ey,ew,eh) in smiles:
            cv2.rectangle (roi_color,(ex,ey),(ex+ew,ey+eh),(0,0,255),2) 
    return frame 
    
#doing some face recognition with the webcam
#video_capture=cv2.VideoCapture(1)#1:external webcam, 0:webcam from computer
#while True:
#    _,frame=video_capture.read()
#    gray=cv2.cvtColor(frame,cv2.COLOR_BGR2GRAY)
#    canvas=detect(gray,frame)
#    cv2.imshow('Video',canvas) #'canvas'=the original image comming from the wc but with the detected rectangle
#    if cv2.waitKey(1) & 0xFF==ord('q'):
#        break        
#video_capture.release()
#cv2.destroyAllWindows()


#doing some object detection on a video
reader=imageio.get_reader('victor.mp4')
fps=reader.get_meta_data()['fps'] #get you fps=#frame/second
writer=imageio.get_writer('victor_output.mp4',fps=fps)

for i,frame in enumerate(reader):
#    frame=detect(frame,net.eval(),transform)
    gray=cv2.cvtColor(frame,cv2.COLOR_BGR2GRAY)
    canvas=detect(gray,frame)
#    cv2.imshow('Video',canvas)
    writer.append_data(canvas)
    print(i)
writer.close()