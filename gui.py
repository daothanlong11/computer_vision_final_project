from __future__ import print_function
from fr_app import *
from imutils.video import VideoStream
import time
import os
import cv2

project_path = os.path.dirname(os.path.realpath(__file__)) + "/"   #path of folfer face_recognition



      
print("[INFO] warming up camera...")
video = VideoStream(0).start()


time.sleep(1.0)

# start the app
pba = app(video,project_path)
pba.root.mainloop()



