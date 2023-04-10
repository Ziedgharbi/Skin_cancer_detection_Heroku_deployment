import numpy as np 
import pandas as pd
import matplotlib.pyplot as plt

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

from flask import Flask ,flash, request, render_template, redirect, url_for
from werkzeug.utils import secure_filename

import urllib.request


project_directory= "C:/Users/pc/Nextcloud/Python/GITHUB/Skin_cancer_detection_Heroku_deployment/"
model_directory=project_directory+"model/"
html_directory=project_directory+'html/'

directory_uploaded_file=project_directory+'image/'
allowed_extension=set(['png', 'jpg', 'jpeg'])

app=Flask(__name__, template_folder=html_directory, static_folder=html_directory+"index_files")

app.secret_key="secret"
app.config['UPLOAD_FOLDER']= directory_uploaded_file

def allowed_file(filename):
    return '.' in filename and filename.rsplit('.',1)[1].lower() in allowed_extension

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

def prediction(img):
    
    model_list=["rgb_h", "gray_L_HE", "gray_L_CLAHE", "gray_HE", "gray","n"]

    rst=[]
    for model in model_list :
        
        im= image_transformation(img, model)
        im=im.reshape(-1,im.shape[0],im.shape[1],im.shape[2])    
        m = keras.models.load_model(model_directory+'best_model_X_'+model+'.h5')
        r=m.predict(im)
        rst.append(np.argmax(r))

    count=[ rst.count(x) for x in range(7)  ] 
    rslt=np.argmax(count)
    

    cancer_class={0:"Melanocytic nevi", 1: "Basal cell carcinoma", 2: "Benign keratosis-like lesions", 
              3 :"Melanoma", 4:"Vascular lesions", 5: "Actinic keratoses", 6: "Dermatofibroma"}
    
    return(cancer_class[rslt])
    
@app.route('/')
def home():
    return render_template('index.html')


@app.route("/" , methods=["GET","POST"])
def predict():
    
    if 'file' not in request.files:
        flash("no file part")
        return redirect(request.url)
    
    file=request.files['file']
    
    if file.filename=="":
        flash('no image selected for uploading')
        return redirect(request.url)
    
    if file and allowed_file(file.filename):
        filename=secure_filename(file.filename)
        file.save(os.path.join(app.config["UPLOAD_FOLDER"], filename))
        x= directory_uploaded_file+filename
        #x=directory_uploaded_file+"ISIC_0024333.jpg"
        
        img=np.asarray(Image.open(x).resize((32,32)))
   
        rs=prediction(img)
        
        flash(str(rs))
        return render_template('index.html', filename=filename)
    else:
        flash("Only png, jpg and gif extension are allowed")
        return redirect(request.url)
   
if __name__ == '__main__':
    app.run()

   
