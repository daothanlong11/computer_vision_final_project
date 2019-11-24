from __future__ import print_function
from sklearn.preprocessing import LabelEncoder
from sklearn.svm import SVC
import pickle
import os

project_path = os.path.dirname(os.path.realpath(__file__))

def learn_svm():
    # load face embeddings
    print("[INFO] loading face embeddings...")
    data = pickle.loads(open(project_path + "/output/embeddings.pickle","rb").read())

    #encode the labels
    print("[INFO] encoding labels...")
    le = LabelEncoder()
    labels = le.fit_transform(data["names"])

    # train the model used to accept the 128-d embeddings of the face and
    # then produce the actual face recognition
    print("[INFO] training model...")
    recognizer = SVC(C=1.0,kernel='linear',probability=True)
    recognizer.fit(data["embeddings"],labels)
    print("training success")

    # write the actual face recognition model to disk
    f = open(project_path + "/output/recognizer_linear.pickle", "wb")
    f.write(pickle.dumps(recognizer))
    f.close()

    # write the label encoder to disk
    f = open(project_path + "/output/le_linear.pickle", "wb")
    f.write(pickle.dumps(le))
    f.close()

    text = "finish learn new model"
    return text