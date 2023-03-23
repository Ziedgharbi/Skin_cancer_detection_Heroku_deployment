import numpy as np 
import pandas as pd
import matplotlib.pyplot as plt

from sklearn.model_selection import train_test_split
from sklearn.preprocessing import LabelEncoder
from sklearn.metrics import confusion_matrix, ConfusionMatrixDisplay, classification_report

import os 
from glob import glob
from PIL import Image
from skimage import color, exposure
from skimage.util import img_as_ubyte
from skimage.filters import rank
from skimage.morphology import disk

import h5py

import tensorflow as tf

from tensorflow import keras

from os import listdir
from os.path import isfile, join

from flask import Flask , request, render_template



project_directory= "C:/Users/pc/Nextcloud/Python/GITHUB/Computer_vision_CNN/"
model_directory=project_directory+"model/"
html_directory=project_directory+'html/'


app=Flask(__name__, template_folder=html_directory, static_folder=html_directory+"/test_files")

cance_classe= {'nv': 0, 'bcc': 1, 'bkl': 2, 'mel': 3, 'vasc': 4, 'akiec': 5, 'df': 6}


def image_transformation(img,trans) :
    
    # no transformation
    
    if trans=="n" :
        img=img
    
    # ---- RGB / Histogram Equalization
    if trans=="rgb_h" :
        hsv = color.rgb2hsv(img)
        hsv[:, :, 2] = exposure.equalize_hist(hsv[:, :, 2])
        img = color.hsv2rgb(hsv)

    # ---- Grayscale

    if trans ==  "gray":
        img=color.rgb2gray(img)
        img=np.expand_dims(img, axis=2)
     

    # ---- Grayscale / Histogram Equalization
    if trans =="gray_HE":
        img=color.rgb2gray(img)
        img=exposure.equalize_hist(img)
        img=np.expand_dims(img, axis=2)


    # ---- Grayscale / Local Histogram Equalization
    if trans=="gray_L_HE":
        img=color.rgb2gray(img)
        img = img_as_ubyte(img)
        img=rank.equalize(img, disk(10))/255.
        img=np.expand_dims(img, axis=2)

            
        
    # ---- Grayscale / Contrast Limited Adaptive Histogram Equalization (CLAHE)
    if trans =="gray_L_CLAHE":
        
        img=color.rgb2gray(img)
        img=exposure.equalize_adapthist(img)
        img=np.expand_dims(img, axis=2)

    return (img)





@app.route('/')
def home():
    return render_template('index.html')






@app.route("/" , methods=["POST"])
def predict():
    
    
    
    
    
    model_list=["rgb_h", "gray_L_HE", "gray_L_CLAHE", "gray_HE", "gray","n"]

    rst={}
    for model in model_list :
        
        im= image_transformation(img, model)
        im=im.reshape(-1,im.shape[0],im.shape[1],im.shape[2])    
        m = keras.models.load_model(model_directory+'best_model_X_'+model+'.h5')
        r=m.predict(im)
        rst[model]=np.argmax(r)

    rst
    
    
    
    
    
    
    
    return render_template('index.html', pred=rst)
   
   
if __name__ == '__main__':
    app.run()

   
