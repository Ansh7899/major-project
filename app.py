from flask import Flask, request, jsonify, render_template, redirect
from CNNmodel import *
# from SVMmodel import *
#svm file
import keras
import numpy as np
import librosa
import pickle

class livePredictions:
    """
    Main class of the application.
    """

    def __init__(self, filename, file):
        """
        Init method is used to initialize the main parameters.
        """
        self.filename = filename
        self.file = file

    def load_model(self):
        """
        Method to load the chosen model.
        :param path: path to your h5 model.
        :return: summary of the model with the .summary() function.
        """
        self.loaded_model = pickle.load(open(self.filename, 'rb'))
        print(self.loaded_model)
        return self.loaded_model

    def makepredictions(self):
        """
        Method to process the files and create your features.
        """
        X, sample_rate = librosa.load(self.file, res_type='kaiser_fast')
        result=[]
        mfccs=np.mean(librosa.feature.mfcc(y=X, sr=sample_rate, n_mfcc=40).T, axis=0)
        result=np.hstack((result, mfccs))
        x = []
        x.append(result)
        predict_x=self.loaded_model.predict(x)
        print(predict_x[0])
        return predict_x[0]

# def SVMtesting(testfile):
#     pred = livePredictions(filename="SVMmodel.sav",file=testfile)
#     pred.load_model()
#     return pred.makepredictions()
    #file main
app = Flask(__name__)

@app.route('/')
def index():
    return render_template('index.html')
                
@app.route('/recognize',methods=['POST'])
def recognize():
    transcript = ""
    transcript1 = ""
    if request.method == "POST":
        print("FORM DATA RECEIVED")

        if "file" not in request.files:
            return redirect(request.url)
        
        file = request.files["file"]
        if file.filename == "":
            return redirect(request.url)
        if file:
            transcript = CNNTesting(file)
        

        file1 = request.files["file1"]
        if file1.filename == "":
            return redirect(request.url)
        if file1:
            pred = livePredictions(filename="SVMmodel.sav",file=file1)
            pred.load_model()
            transcript1 = pred.makepredictions()

    return render_template('index.html', transcript=transcript , transcript1 = transcript1)
    

if __name__ == "__main__":
    app.run(debug=True, threaded=True)