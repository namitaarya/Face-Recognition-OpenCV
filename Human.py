#imported modules
import cv2 as cv
import numpy as np

#imported model data through classifier 
classify= cv.CascadeClassifier("D:\My Folders\Projects\Outlining human\haarcascade_frontalface_default.xml")
smile_cascade = cv.CascadeClassifier("D:\My Folders\Projects\Outlining human\haarcascade_smile.xml")
eyes_cascade= cv.CascadeClassifier("D:\My Folders\Projects\Outlining human\haarcascade_eye.xml")

#starts web cam indexed 0
img=cv.VideoCapture(0)
while True:
    ret, frame= img.read()
    if ret:
       #detects face and stores 4 variables 
        faces= classify.detectMultiScale(frame,1.3,5)

       #img2 stores face part since we have 4 variables 
        for x,y,w,h in faces:
            img2= frame[y:y+h, x:x+w] 

            #sketches a rectangle around the coordinates                       
            cv.rectangle(frame,(x,y),(x+w,y+h),(255,0,0),3)

            #detects smile in a specific part of image i.e face or img2           
            smiles = smile_cascade.detectMultiScale(img2,1.8, 20)
          
            #sketches a rectange around smile
            for sx,sy,sw,sh in smiles:
                smile=cv.rectangle(img2, (sx, sy), ((sx + sw), (sy + sh)), (0, 0, 255), 3)

            #finds the coordinates for eyes in the img2 i.e face
            eyes= eyes_cascade.detectMultiScale(img2)

            #sketches rectangle around eyes
            for ex,ey,ew,eh in eyes:
                cv.rectangle(img2,(ex,ey),((ex+ew),(ey+eh)),(0,255,0),2)


        cv.imshow("Thats your face",frame)
        
        if cv.waitKey(5)==ord('q'):
            break

img.release()
cv.destroyAllWindows()
