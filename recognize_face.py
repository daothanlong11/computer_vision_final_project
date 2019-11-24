from imutils.video import VideoStream
from imutils.video import FPS
import numpy as np
import imutils
import pickle
import time
import cv2
import os
from fr_app import *

project_path = os.path.dirname(os.path.realpath(__file__))

# load face detector model
print("[INFO] loading face detector...")
protoPath = project_path + "/face_detection_model/deploy.prototxt"
modelPath = project_path + "/face_detection_model/res10_300x300_ssd_iter_140000.caffemodel"
detector = cv2.dnn.readNetFromCaffe(protoPath, modelPath)

# load face embedding initial model 
print("[INFO] loading face recognizer...")
embeedingPath = project_path + "/openface_nn4.small2.v1.t7"
embedder = cv2.dnn.readNetFromTorch(embeedingPath)

# load the actual face recognition model along with the label encoder
recognizer = pickle.loads(open(project_path + "/output/recognizer_linear.pickle",'rb').read())
le = pickle.loads(open(project_path + "/output/le_linear.pickle",'rb').read())


# initialize the video stream, then allow the camera sensor to warm up
print("[INFO] starting video stream...")
vs = VideoStream(src=0).start()
time.sleep(2.0)

# start the FPS throughput estimator
fps = FPS().start()

# start the app
pba = app(vs,"/home/l/Documents/code/python/test_code/")
pba.root.mainloop()

while True:
    frame = vs.read()

    frame = imutils.resize(frame, width=600)
    (h, w) = frame.shape[:2]

    # construct a blob from the image
    imageBlob = cv2.dnn.blobFromImage(
		cv2.resize(frame, (300, 300)), 1.0, (300, 300),
		(104.0, 177.0, 123.0), swapRB=False, crop=False)

    # apply OpenCV's deep learning-based face detector to localize
	# faces in the input image
    detector.setInput(imageBlob)
    detections = detector.forward()

    for i in range(0,detections.shape[2]):
        confidence = detections[0,0,i,2]

        if (confidence > 0.5):
            # compute the (x, y)-coordinates of the bounding box for
			# the face
            box = detections[0, 0, i, 3:7] * np.array([w, h, w, h])
            (startX, startY, endX, endY) = box.astype("int")

            # extract the face ROI and grab the ROI dimensions
            face = frame[startY:endY, startX:endX]
            (fH, fW) = face.shape[:2]

            # ensure the face width and height are sufficiently large
            if fW < 20 or fH < 20:
                continue

            # construct a blob for the face ROI, then pass the blob
			# through our face embedding model to obtain the 128-d
			# quantification of the face
            faceBlob = cv2.dnn.blobFromImage(face, 1.0 / 255,(96, 96), (0, 0, 0), swapRB=True, crop=False)
                
            # 128-d ouput
            embedder.setInput(faceBlob)
            vec = embedder.forward()

            #perform classification 
            preds = recognizer.predict_proba(vec)[0]
            j = np.argmax(preds)
            proba = preds[j]
            name = le.classes_[j]

            # draw the bounding box of the face along with the
			# associated probability
            text = "{}: {:.2f}%".format(name, proba * 100)
            y = startY - 10 if startY - 10 > 10 else startY + 10
            cv2.rectangle(frame, (startX, startY), (endX, endY),
				(0, 0, 255), 2)
            cv2.putText(frame, text, (startX, y),
				cv2.FONT_HERSHEY_SIMPLEX, 0.45, (0, 0, 255), 2)

    # update the FPS counter
    fps.update()

    # show the output frame
    cv2.imshow("Frame", frame)
    key = cv2.waitKey(1) & 0xFF

	# if the `q` key was pressed, break from the loop
    if key == ord("q"):
        break

# stop the timer and display FPS information
fps.stop()
print("[INFO] elasped time: {:.2f}".format(fps.elapsed()))
print("[INFO] approx. FPS: {:.2f}".format(fps.fps()))

# do a bit of cleanup
cv2.destroyAllWindows()
vs.stop()
