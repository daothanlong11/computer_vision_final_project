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
import pickle
import numpy as np
import pandas as pd 
from imutils import paths
import time

from face_alignment import face_align
from extract_embedding import extract_embd
from train_svm_model import learn_svm

project_path = os.path.dirname(os.path.realpath(__file__)) #path of folfer face_recognition

class app:
    def __init__(self,video,outputPath):
        self.video = video
        self.outputPath = outputPath
        #self.frame1 = None
        self.frame2 = None
        self.name = None
        self.path = outputPath
        #self.thread1 = None
        self.thread2 = None
        self.stopEvent = None
        self.submit = None

        
        # initialize the root window and image panel
        self.root = tk.Tk()
        

        #self.panel1 = None
        self.panel2 = None
        self.flag = True

        # learn info
        self.text = None

        #attendace info
        self.name_att = []
        self.mssv_att = []
        self.date_att = []
        
        # Time
        self.time = datetime.datetime.now()

        # count frame
        self.i = 0
        
        # create a button, that when pressed, will take the current
		# frame and save it to file
        btn3 = tk.Button(self.root,text='stop scan',command=self.stop_scan)
        btn3.pack(side='bottom',fill='both',expand='yes',padx=10,pady=10)
        
        btn2 = tk.Button(self.root,text='start scan',command=self.start_scan)
        btn2.pack(side='bottom',fill='both',expand='yes',padx=10,pady=10)

        btn1 = tk.Button(self.root,text='learn',command=self.start_learn)
        btn1.pack(side='bottom',fill='both',expand='yes',padx=10,pady=10)

        btn4 = tk.Button(self.root,text='attendance file',command=self.attendance_file)
        btn4.pack(side='bottom',fill='both',expand='yes',padx=10,pady=10)

        btn5 = tk.Button(self.root,text='create sound',command=self.create_sound)
        btn5.pack(side='bottom',fill='both',expand='yes',padx=10,pady=10)        
        
        # create text input for name and date
        lbl1 = tk.Label(self.root, text="NAME")
        lbl1.pack(side='bottom')
        self.entry_id1 = StringVar()
        entry1 = tk.Entry(self.root, textvariable=self.entry_id1,justify='center')
        entry1.pack(side='bottom',expand='yes',padx=10,pady=10)

        lbl2 = tk.Label(self.root, text="MSSV")
        lbl2.pack(side='bottom')
        self.entry_id2 = StringVar()
        entry2 = tk.Entry(self.root, textvariable=self.entry_id2,justify='center')
        entry2.pack(side='bottom',expand='yes',padx=10,pady=10)

        lbl3 = tk.Label(self.root, text="STATUS")
        lbl3.pack(side='bottom')
        self.text = tk.Text(self.root, height=2, width=30)
        self.text.pack(side='bottom')
        
                
        
        
    
        # start a thread that constantly pools the video sensor for
		# the most recently read frame
        self.stopEvent = threading.Event() 
        self.thread2 = threading.Thread(target=self.videoLoop2,args=())
        self.thread2.start()
        self.thread3 = threading.Thread(target=self.learn_new_face,args=())
        

        # set a callback to handle when the window is closed
        self.root.wm_title("Face Recognition System")
        self.root.wm_protocol("WM_DELETE_WINDOW", self.onClose)

  

    def videoLoop2(self): 
        try:
            while not self.stopEvent.is_set():
                if (self.flag == True):
                    self.frame2 = self.video.read()
                    self.frame2 = imutils.resize(self.frame2,width=600)

                    image = cv2.cvtColor(self.frame2, cv2.COLOR_BGR2RGB)
                    image = Image.fromarray(image)
                    image = ImageTk.PhotoImage(image)

                    
                    if (self.submit == True):
                        if not os.path.exists(project_path+"/dataset/{}".format(self.name)):
                            os.mkdir(project_path+"/dataset/{}".format(self.name))

                        s = 0
                        if len(self.name_att) == 0:
                            self.name = self.entry_id1.get()
                            mssv = self.entry_id2.get()
                            date = self.time.strftime("%d-%m-%Y_%H:%M:%S")
                            self.name_att.append(self.name)
                            self.mssv_att.append(mssv)
                            self.date_att.append(date)
                    
                        else:
                            for n in self.name_att:
                                if self.name == n:
                                    s = 1
                            
                            if s == 0:
                                self.name = self.entry_id1.get()
                                mssv = self.entry_id2.get()
                                date = self.time.strftime("%d-%m-%Y_%H:%M:%S")
                                self.name_att.append(self.name)
                                self.mssv_att.append(mssv)
                                self.date_att.append(date)

                        if (self.i%30 == 0):
                            self.name = self.entry_id1.get()
                            imagePaths = list(paths.list_images(self.path + "dataset/{}/".format(self.name)))           
                            if len(imagePaths) < 1:
                                cv2.imwrite(self.path+"dataset/{}/0.png".format(self.name),self.frame2.copy())
                                print("save new dataset success")
                            else:
                                cv2.imwrite(self.path+"dataset/{}/{}.png".format(self.name,len(imagePaths)),self.frame2.copy())
                                print("save new dataset success")

                        self.i+=1 
                    
                    
                    # if the panel is not None, we need to initialize it
                    if self.panel2 is None:
                        self.panel2 = tk.Label(image=image)
                        self.panel2.image = image
                        self.panel2.pack(side='left',padx=10,pady=10)

                    # otherwise, simply update the panel
                    else:
                        self.panel2.configure(image=image)
                        self.panel2.image = image

                elif (self.flag == False):
                    self.frame2 = cv2.imread(project_path + "/bk.jpg")
                    self.frame2 = imutils.resize(self.frame2,width=200)

                    image = cv2.cvtColor(self.frame2, cv2.COLOR_BGR2RGB)
                    image = Image.fromarray(image)
                    image = ImageTk.PhotoImage(image)

                    if self.panel2 is None:
                        self.panel2 = tk.Label(image=image)
                        self.panel2.image = image
                        self.panel2.pack(side='left',padx=10,pady=10)

                    # otherwise, simply update the panel
                    else:
                        self.panel2.configure(image=image)
                        self.panel2.image = image   
                
        
        except RuntimeError :
            print("[INFO] caught a RuntimeError")

       
    def start_scan(self):
        self.submit = True
    
    def stop_scan(self):
        self.submit = False
        self.i = 0
        


    def attendance_file(self):
        input_line = {'name':self.name_att,'mssv':self.mssv_att,'attendance_time':self.date_att}
        df2 = pd.DataFrame(input_line)

        
        df1 = pd.read_csv(project_path+"/output/danh_sach_diem_danh.csv",index_col=0)
        df1 = df1.append(df2,ignore_index = True,sort=False)
        df1.to_csv(project_path+"/output/danh_sach_diem_danh.csv")
        
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
        with io.open(project_path+"/output/danh_sach_diem_danh.csv", "r", newline="") as csv_file:
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

    def learn_new_face(self):
        self.text.insert(tk.END,"face alignment procesing...")
        status1 = face_align()
        self.text.delete(1.0,tk.END)
        self.text.insert(tk.END,status1)
        time.sleep(1.0)
        self.text.delete(1.0,tk.END)
        
        self.text.insert(tk.END,"extract embdding procesing...")
        status2 = extract_embd()
        self.text.delete(1.0,tk.END)
        self.text.insert(tk.END,status2)
        time.sleep(1.0)
        self.text.delete(1.0,tk.END)
        

        self.text.insert(tk.END,"learn procesing...")
        status3 = learn_svm()
        self.text.delete(1.0,tk.END)
        time.sleep(1.0)
        self.text.insert(tk.END,status3)

        self.flag = True # return to normal mode

    def create_sound(self):
        from gtts import gTTS 
        from pygame import mixer
        from pydub import AudioSegment

        for name in self.name_att:
            n = name.split(" ")[-1]
            language = 'vi'
            myobj = gTTS(text=n, lang=language, slow=False)  
            myobj.save(project_path+"/sound/{}.mp3".format(n)) 

            # files                                                                         
            src = project_path + "/sound/{}.mp3".format(n)
            dst = project_path + "/sound/{}.wav".format(n)

            # convert wav to mp3                                                            
            sound = AudioSegment.from_mp3(src)
            sound.export(dst, format="wav")

        
    def start_learn(self):
        self.flag = False # turn on learn mode
        self.thread3.start()

    def onClose(self):
        # set the stop event, cleanup the camera, and allow the rest of
        # the quit process to continue
        print("[INFO] closing...")
        self.stopEvent.set()
        self.video.stop()
        self.root.quit()

    



            