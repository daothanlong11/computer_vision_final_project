from __future__ import print_function
import cv2
import os

project_path = os.path.dirname(os.path.realpath(__file__))
name = "long"
i = 0
cap = cv2.VideoCapture(0)

while True:
    ret,frame = cap.read()

    cv2.imshow("frame",frame)
   
    key = cv2.waitKey(1) & 0xFF

	# if the `q` key was pressed, break from the loop
    if key == ord('s'):
        cv2.imwrite(project_path + "/dataset" + "/" + name + "/"  + "{}.png".format(i),frame)
        i+=1

    
cv2.destroyAllWindows()
