################ Function to detect Landmarks using dlib #####################

# =============================================================================
# def showImage(img):
#     #imgplot = plt.imshow(img)
#     plt.show(plt.imshow(img))
#     
# def loadModels(predictorModelPath):
#     global detector
#     global predictor
#     detector = dlib.get_frontal_face_detector()
#     predictor = dlib.shape_predictor("shape_predictor_68_face_landmarks.dat")    
#     print("dlib landmarks model loaded")
#     
# def loadImagesForLandmarkDetection(img):
#     img=mpimg.imread(img)
#     showImage(img)
#     return img
# 
# def drawLandmarks(landmarks,img):
#     for n in range(0, 68):
#         x = landmarks.part(n).x
#         y = landmarks.part(n).y
#         cv2.circle(img, (x, y), 1, (255, 0, 0), -1)  
#         showImage(img)
#         
# def detectLandmarks(img)  :
#     img = loadImagesForLandmarkDetection(img)
#     faces = detector(img)
#     for face in faces:
#         x1 = face.left()
#         y1 = face.top()
#         x2 = face.right()
#         y2 = face.bottom()
#         #cv2.rectangle(frame, (x1, y1), (x2, y2), (0, 255, 0), 3)
#     
#         landmarks = predictor(img, face)
#         drawLandmarks(landmarks,img)
#     
#     
#     x1 = landmarks.part(6).x  
#     y1 = landmarks.part(2).y
#     
#     x2 = landmarks.part(10).x
#     y2 = landmarks.part(6).y
#     h=y2-y1
#     w=x2-x1   
#     
#     #print("x1 :" +str(x1) + ", y1 :" +str(y1))
#     #print("x2 :" +str(x2) + ", y2 :" +str(y2))
#     #print("width :" +str(w) + ", height :" +str(h))
#     #print("width :" +str(w) + ", height :" +str(h))
#     #print("x+w :" +str(x1+w) + ", y+h :" +str(y1+h))
#     crop_img = img[y1:y1+h, x1:x1+w]
#     #showImage(img)
#     showImage(crop_img)
#     return crop_img
# =============================================================================