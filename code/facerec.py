import cv2
import numpy as np
import face_recognition

imgElon = face_recognition.load_image_file('imageBasic/ElonMusk.jpg')
imgElon = cv2.cvtColor(imgElon,cv2.COLOR_BGR2RGB)
imgTest = face_recognition.load_image_file('imageBasic/ElonTest.jpg')
imgTest = cv2.cvtColor(imgTest,cv2.COLOR_BGR2RGB)



faceLocation = face_recognition.face_locations(imgElon)[0]
encodeElon = face_recognition.face_encodings(imgElon)[0]
cv2.rectangle(imgElon,(faceLocation[3],faceLocation[0]),(faceLocation[1],faceLocation[2],),(255,0,0),2)

faceLocationTest = face_recognition.face_locations(imgTest)[0]
encodeTest = face_recognition.face_encodings(imgTest)[0]
cv2.rectangle(imgTest,(faceLocationTest[3],faceLocationTest[0]),(faceLocationTest[1],faceLocationTest[2],),(255,0,0),2)



results = face_recognition.compare_faces([encodeElon],encodeTest)
faceDistance = face_recognition.face_distance([encodeElon],encodeTest)
print(results)
print(faceDistance)
cv2.putText(imgTest,f'{results} {faceDistance}', (50,50),cv2.FONT_HERSHEY_COMPLEX,1,(0,0,255),2)

cv2.imshow('Elon Musk',imgElon)
cv2.imshow('Elon Test',imgTest)

cv2.waitKey(0)