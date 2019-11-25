from __future__ import print_function
from PIL import Image
from PIL import ImageTk
import tkinter as tk
from tkinter import ttk, StringVar
import threading
import datetime
import imutils
import cv2
import os
import numpy as np
import pickle
from imutils import paths
import pandas as pd 
from pygame import mixer
from multiprocessing import Process

project_path = os.path.dirname(os.path.realpath(__file__)) #path of folfer face_recognition


# load face detector model
print("[INFO] loading face detector...")
protoPath = project_path + "/face_detection_model/deploy.prototxt"
modelPath = project_path + "/face_detection_model/res10_300x300_ssd_iter_140000.caffemodel"
detector = cv2.dnn.readNetFromCaffe(protoPath, modelPath)

# load face embedding initial model 
print("[INFO] loading face recognizer...")
embeedingPath = project_path + "/face_detection_model/openface_nn4.small2.v1.t7"
embedder = cv2.dnn.readNetFromTorch(embeedingPath)

# load the actual face recognition model along with the label encoder
recognizer = pickle.loads(open(project_path + "/output/recognizer_linear.pickle",'rb').read())
le = pickle.loads(open(project_path + "/output/le_linear.pickle",'rb').read())



class app:
    def __init__(self,video,outputPath):
        self.video = video
        self.outputPath = outputPath
        self.frame1 = None
        #self.frame2 = None
        self.name = None
        self.path = outputPath
        self.thread1 = None
        self.stopEvent = None
        self.flag = False
        
        # initialize the root window and image panel
        self.root = tk.Tk()
        self.panel1 = None

        # attendance info
        self.name_att = []
        self.date_att = []
        self.status_att = []


        # Time
        self.time = datetime.datetime.now()

        # create a button, that when pressed, will take the current
		# frame and save it to file
        btn2 = tk.Button(self.root,text='start',command=self.start_app)
        btn2.pack(side='bottom',fill='both',expand='yes',padx=10,pady=10)

        btn4 = tk.Button(self.root,text='open attendance',command=self.attendance_file)
        btn4.pack(side='bottom',fill='both',expand='yes',padx=10,pady=10)

        # create text input for name and date
        lbl1 = tk.Label(self.root, text="Attendace Time")
        lbl1.pack(side='bottom')
        self.entry_id1 = StringVar()
        entry1 = tk.Entry(self.root, textvariable=self.entry_id1,justify='center')
        entry1.pack(side='bottom',expand='yes',padx=10,pady=10)
        
         
        # start a thread that constantly pools the video sensor for
		# the most recently read frame
        self.stopEvent = threading.Event()
        self.thread1 = threading.Thread(target=self.videoLoop1,args=())
        self.thread2 = threading.Thread(target=self.speak,args=())
        #self.process1 = Process(target=speak, args=())
             
        # set a callback to handle when the window is closed
        self.root.wm_title("Face Recognition System")
        self.root.wm_protocol("WM_DELETE_WINDOW", self.onClose)

    def start_app(self):
        self.thread1.start()
        self.thread2.start()
        

    def videoLoop1(self):
        try:
            while not self.stopEvent.is_set():       
                
                self.frame1 = self.video.read()
                self.frame1 = imutils.resize(self.frame1,width=600)

                (h, w) = self.frame1.shape[:2]

                # construct a blob from the image
                imageBlob = cv2.dnn.blobFromImage(
                    cv2.resize(self.frame1, (300, 300)), 1.0, (300, 300),
                    (104.0, 177.0, 123.0), swapRB=False, crop=False)

                # apply OpenCV's deep learning-based face detector to localize
                # faces in the input image
                detector.setInput(imageBlob)
                detections = detector.forward()

                
                for i in range(0,detections.shape[2]):
                    confidence = detections[0,0,i,2]

                    if (confidence > 0.7):
                        # compute the (x, y)-coordinates of the bounding box for
                        # the face
                        box = detections[0, 0, i, 3:7] * np.array([w, h, w, h])
                        (startX, startY, endX, endY) = box.astype("int")

                        # extract the face ROI and grab the ROI dimensions
                        face = self.frame1[startY:endY, startX:endX]
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
                        self.name = name
                        # write name and time to csv file
                        s = 0

                        if len(self.name_att) == 0:
                            attendance_time = self.time.strftime("%d-%m-%Y_%H:%M:%S")
                            date1 = attendance_time.split("_")[-1]
                            date2 = self.entry_id1.get()
                            if (date1 >= date2):
                                self.status_att.append('late')
                            else:
                                self.status_att.append('on time')
                            self.name_att.append(name)
                            self.date_att.append(attendance_time)
                            
                            self.flag = True
                            
                            
                        else:

                            for n in self.name_att:
                                if name == n:
                                    s = 1
                            
                            if s == 0:
                                attendance_time = self.time.strftime("%d-%m-%Y_%H:%M:%S")
                                date1 = attendance_time.split("_")[-1]
                                date2 = self.entry_id1.get()
                                if (date1 >= date2):
                                    self.status_att.append('late')
                                else:
                                    self.status_att.append('on time')
                                self.name_att.append(name)
                                self.date_att.append(attendance_time)
                                
                                self.flag = True
                                
                                
                        # draw the bounding box of the face along with the
                        # associated probability
                        text = "{}: {:.2f}%".format(name.split(" ")[-1], proba * 100)
                        y = startY - 10 if startY - 10 > 10 else startY + 10
                        cv2.rectangle(self.frame1, (startX, startY), (endX, endY),
                            (0, 0, 255), 2)
                        cv2.putText(self.frame1, text, (startX, y),
                        cv2.FONT_HERSHEY_SIMPLEX, 0.45, (0, 0, 255), 2)
                    self.flag = False

                image = cv2.cvtColor(self.frame1, cv2.COLOR_BGR2RGB)
                image = Image.fromarray(image)
                image = ImageTk.PhotoImage(image)
                

                # if the panel is not None, we need to initialize it
                if self.panel1 is None:
                    self.panel1 = tk.Label(image=image)
                    self.panel1.image = image
                    self.panel1.pack(side='left',padx=10,pady=10)
                    

                # otherwise, simply update the panel
                else:
                    self.panel1.configure(image=image)
                    self.panel1.image = image
        
        except RuntimeError:
            print("[INFO] caught a RuntimeError")   

    def attendance_file(self):
        input_line = {'name':self.name_att,'date':self.date_att,'status':self.status_att}
        df = pd.DataFrame(input_line)

        with open(project_path+"/output/diemdanh.csv",'a') as f:
            df.to_csv(project_path+"/output/diemdanh.csv")
        f.close()

        
        import csv
        import io
        from tkmagicgrid import MagicGrid
        
        # Create a root window
        root = tk.Tk()

        # Create a MagicGrid widget
        grid = MagicGrid(root)
        grid.pack(side="top", expand=1, fill="both")

        # Display the contents of some CSV file
        # (note this is not a particularly efficient viewer)
        with io.open(project_path+"/output/diemdanh.csv", "r", newline="") as csv_file:
            reader = csv.reader(csv_file)
            parsed_rows = 0
            for row in reader:
                if parsed_rows == 0:
                    # Display the first row as a header
                    grid.add_header(*row)
                else:
                    grid.add_row(*row)
                parsed_rows += 1

        # Start Tk's event loop
        root.mainloop()

    def speak(self):
        try:
            mixer.init()
            while not self.stopEvent.is_set():      
                if (self.flag == True):
                    n = self.name.split(" ")[-1]
                    dst = project_path + "/sound/{}.wav".format(n)
                    mixer.music.load(dst)
                    mixer.music.play()
                
                print("0")

        except RuntimeError:
            print("[INFO] caught a RuntimeError") 
        

    def onClose(self):
        # set the stop event, cleanup the camera, and allow the rest of
        # the quit process to continue
        print("[INFO] closing...")
        self.stopEvent.set()
        self.video.stop()
        self.root.quit()

