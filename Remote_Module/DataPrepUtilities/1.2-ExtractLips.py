import cv2
import dlib
import matplotlib.pyplot as plt
import matplotlib.image as mpimg



predictorModelPath = r"/home/ahmad/Avatar/MachineLearning/dlib_test/shape_predictor_68_face_landmarks.dat"

testImage = r"/home/ahmad/Avatar/MachineLearning/dlib_test/test4.jpg"
detector = dlib.get_frontal_face_detector()
predictor = dlib.shape_predictor("shape_predictor_68_face_landmarks.dat")

print("dlib landmarks model loaded")
img=mpimg.imread(testImage)
imgplot = plt.imshow(img)
plt.show()
#cv2.imshow("img", img)
#gray = cv2.cvtColor(img, cv2.COLOR_BGR2GRAY)

faces = detector(img)
for face in faces:
    x1 = face.left()
    y1 = face.top()
    x2 = face.right()
    y2 = face.bottom()
    #cv2.rectangle(frame, (x1, y1), (x2, y2), (0, 255, 0), 3)

    landmarks = predictor(img, face)

# =============================================================================
#     for n in range(0, 68):
#         x = landmarks.part(n).x
#         y = landmarks.part(n).y
#         cv2.circle(img, (x, y), 1, (255, 0, 0), -1)
# =============================================================================

x1 = landmarks.part(6).x  
y1 = landmarks.part(2).y

x2 = landmarks.part(10).x
y2 = landmarks.part(6).y

print("x1 :" +str(x1) + ", y1 :" +str(y1))
print("x2 :" +str(x2) + ", y2 :" +str(y2))

#print("width :" +str(w) + ", height :" +str(h))

h=y2-y1
w=x2-x1


print("width :" +str(w) + ", height :" +str(h))
print("x+w :" +str(x1+w) + ", y+h :" +str(y1+h))
crop_img = img[y1:y1+h, x1:x1+w]
#cv2.imshow("cropped", crop_img)

imgplot = plt.imshow(img)
plt.show()

imgplot = plt.imshow(crop_img)
plt.show()