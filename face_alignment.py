from __future__ import print_function
from imutils.face_utils import FaceAligner
from imutils.face_utils import rect_to_bb
import imutils
import dlib
import cv2
import os
from imutils import paths

project_path = os.path.dirname(os.path.realpath(__file__))

def face_align():
    detector = dlib.get_frontal_face_detector()
    predictor = dlib.shape_predictor(project_path + "/face_alignment_model/shape_predictor_68_face_landmarks.dat")
    fa = FaceAligner(predictor,desiredFaceWidth=256)

    imagePaths = list(paths.list_images(project_path + "/dataset/"))

    for (i,imagePath) in enumerate(imagePaths):

        name_file = imagePath.split(os.path.sep)[-1]
        name_folder = imagePath.split('/')[-2]
        name = name_file.split('.')[-2]
        tale = name_file.split('.')[-1]
        
        image = cv2.imread(imagePath)
        image = imutils.resize(image,width=800)
        gray = cv2.cvtColor(image,cv2.COLOR_BGR2GRAY)

        #cv2.imshow("Input", image)
        rects = detector(gray,2)

        for rect in rects:
            # extract the ROI of the *original* face, then align the face
            # using facial landmarks
            (x,y,h,w) = rect_to_bb(rect)
            #faceOrig = imutils.resize(image[y:y+h,x:x+w],width=256)
            faceAligned = fa.align(image,gray,rect)
            cv2.imwrite("/home/l/Documents/code/python/face_recognition/dataset/{}/{}.{}".format(name_folder,name,tale),faceAligned)
            #cv2.imshow("Original", faceOrig)
            #cv2.imshow("Aligned", faceAligned)
            #cv2.waitKey(0)

    #print("success aligned all image in datasets")
    text = "finish face alignment"   
    return text