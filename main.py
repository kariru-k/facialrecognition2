# imports
import cv2
import face_recognition

anchorImg = face_recognition.load_image_file('lfw/Angela_Merkel/Angela_Merkel_0001.jpg')
anchorImg = cv2.cvtColor(anchorImg, cv2.COLOR_BGR2RGB)

testImg = face_recognition.load_image_file('lfw/Angela_Mascia-Frye/Angela_Mascia-Frye_0001.jpg')
testImg = cv2.cvtColor(testImg, cv2.COLOR_BGR2RGB)

#detecting the anchor face
faceLoc = face_recognition.face_locations(anchorImg)[0]
encodedAnchor = face_recognition.face_encodings(anchorImg)[0]

#rectangle around the anchor face
cv2.rectangle(anchorImg, (faceLoc[3], faceLoc[0]), (faceLoc[1], faceLoc[2]), (255, 8, 255))

#detecting the test face
faceLoctest = face_recognition.face_locations(testImg)[0]
encodedtest = face_recognition.face_encodings(testImg)[0]

#rectangle around the test face
cv2.rectangle(testImg, (faceLoctest[3], faceLoctest[0]), (faceLoctest[1], faceLoctest[2]), (255, 8, 255))

#checking if encodings are similar
results = face_recognition.compare_faces([encodedAnchor], encodedtest)
faceDis = face_recognition.face_distance([encodedAnchor], encodedtest)
print(results, faceDis)
cv2.putText(testImg,f'{results} {round(faceDis[0],2)}',(50,50),cv2.FONT_HERSHEY_COMPLEX,1,(0,0,255),2)

cv2.imshow('Angela Merkel', anchorImg)
cv2.imshow('Angela Merkel tester', testImg)
cv2.waitKey(0)


