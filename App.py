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

tf.random.set_seed(1)
np.random.seed(1)
project_directory= "C:/Users/pc/Nextcloud/Python/GITHUB/Computer_vision_CNN/"
data_directory=project_directory+"data/"
image_directory= data_directory +"images/"
data_transorfmed_directory=project_directory+'transformed_data/'
model_directory=project_directory+"model/"


isExist = os.path.exists(data_transorfmed_directory)
if not isExist:

   # Create a new directory because it does not exist
   os.makedirs(data_transorfmed_directory)
   print("The new directory is created!")

df=pd.read_csv(data_directory+"HAM10000_metadata.csv")
df.columns

df['image_path']= [os.path.join(image_directory+str(name)+'.jpg') for name in df["image_id"]]

"""----- if you don't have so much ressources you can limit data ---------"""

frac = 1  # take 30% of data only 

data=df.sample(frac=frac, random_state=1)

data["image_array"]= data['image_path'].map(lambda x: np.asarray(Image.open(x).resize((32,32))))


# plot random picture
n_sample=15
random=np.random.randint(0,data.shape[0],n_sample)

plt.subplots (3,5, figsize=(5,5))

for i in range(15):
    plt.subplot(3,5,i+1)
    plt.imshow(data.image_array.iloc[random[i]])
    plt.title(data.dx.iloc[i])
plt.show()


# some descreptive statistic of classes 
data.dx.value_counts().plot(kind='pie') # data are unbalanced, we should balance it


# some manuel encoding for taregt variable or use label encoder of sklearn
""" nv : Melanocytic nevi  / mel : Melanoma    /
  bkl : Benign keratosis-like lesions  / bcc : Basal cell carcinoma
  akiec : Actinic keratoses   / vas : Vascular lesions /   df :  Dermatofibroma     """
data.dx.unique()

classe=dict(zip(data.dx.unique(), range (len(data.dx.unique()))   ))
class_cancer=list(classe.keys())

data["target"]=data.dx.map(lambda x : classe[x]  )


"""------- do somes transformation on image and for each we train a model and voting for rslt ---------"""

# first balance data

data.target.value_counts()

n_sample=500 # you can choose other number : see unique value effectifs first

data_balanced=pd.DataFrame()
for i in data.target.unique() :
    d=data[data.target == i]
    temp= d.sample(n=n_sample, replace=True)
    data_balanced=pd.concat( [data_balanced, temp])
    
data_balanced=data

X=np.asarray(data_balanced.image_array) 
X=X/255
y=data_balanced.target

y.value_counts()
len(np.unique(y))
y=tf.keras.utils.to_categorical(y, num_classes=7)


# transformation : 5 transformation 


def image_transformation(X,y,data_transorfmed_directory) :
    
    # no transofrmation
    X_n=[]
    for img in X:
        X_n.append(img)   

    with h5py.File(data_transorfmed_directory+"X_n.h5", "w") as f:
        f.create_dataset("X", data=X_n)
        f.create_dataset("y", data=y)
    
    # ---- RGB / Histogram Equalization
    X_rgb_h=[]
    for img in X:
        hsv = color.rgb2hsv(img)
        hsv[:, :, 2] = exposure.equalize_hist(hsv[:, :, 2])
        img = color.hsv2rgb(hsv)
        X_rgb_h.append(img)   

    with h5py.File(data_transorfmed_directory+"X_rgb_h.h5", "w") as f:
        f.create_dataset("X", data=X_rgb_h)
        f.create_dataset("y", data=y)

    #X_rgb_h[1].shape  ## dx,dy, dz

    # ---- Grayscale
    X_gray=[]
    for img in X:
        img=color.rgb2gray(img)
        img=np.expand_dims(img, axis=2)
        X_gray.append(img)
        
    with h5py.File(data_transorfmed_directory+"X_gray.h5", "w") as f:
        f.create_dataset("X", data=X_gray)
        f.create_dataset("y", data=y)
              
    #X_gray[1].shape ### dx,dy


    # ---- Grayscale / Histogram Equalization
    X_gray_HE=[]
    for img in X:
        img=color.rgb2gray(img)
        img=exposure.equalize_hist(img)
        img=np.expand_dims(img, axis=2)
        X_gray_HE.append(img)     ###################

    with h5py.File(data_transorfmed_directory+"X_gray_HE.h5", "w") as f:
        f.create_dataset("X", data=X_gray_HE)
        f.create_dataset("y", data=y)

    #X_gray_HE[1].shape  # dx,dy

    # ---- Grayscale / Local Histogram Equalization
    X_gray_L_HE=[]
    for img in X:
        img=color.rgb2gray(img)
        img = img_as_ubyte(img)
        img=rank.equalize(img, disk(10))/255.
        img=np.expand_dims(img, axis=2)
        X_gray_L_HE.append(img)
        
    with h5py.File(data_transorfmed_directory+"X_gray_L_HE.h5", "w") as f:
        f.create_dataset("X", data=X_gray_L_HE)
        f.create_dataset("y", data=y)
      
    #X_gray_L_HE[1].shape  # dx,dy      
        
    # ---- Grayscale / Contrast Limited Adaptive Histogram Equalization (CLAHE)
    X_gray_L_CLAHE=[]
    for img in X:
        img=color.rgb2gray(img)
        img=exposure.equalize_adapthist(img)
        img=np.expand_dims(img, axis=2)
        X_gray_L_CLAHE.append(img)

    with h5py.File(data_transorfmed_directory+"X_gray_L_CLAHE.h5", "w") as f:
        f.create_dataset("X", data=X_gray_L_CLAHE)
        f.create_dataset("y", data=y)

    #X_gray_L_CLAHE[1].shape  # dx,dy



image_transformation(X,y,data_transorfmed_directory)

# create model 
def create_model (dx,dy,dz) :
    model = keras.models.Sequential()
    
    model.add( keras.layers.Conv2D(256, (3,3), activation='relu', input_shape=(dx,dy,dz)))
    model.add( keras.layers.MaxPooling2D((2, 2)))
    model.add( keras.layers.Dropout(0.3))

    model.add( keras.layers.Conv2D(128, (3, 3), activation='relu'))
    model.add( keras.layers.MaxPooling2D((2, 2)))
    model.add( keras.layers.Dropout(0.3))
    
    model.add( keras.layers.Conv2D(64, (3, 3), activation='relu'))
    model.add( keras.layers.MaxPooling2D((2, 2)))
    model.add( keras.layers.Flatten()) 
 
    model.add( keras.layers.Dense(32, activation='relu'))
    
    model.add( keras.layers.Dense(7, activation='softmax'))
    
    return model
 

## train model with data generator from keras 

transformation=["X_rgb_h", "X_gray", "X_gray_HE", "X_gray_L_HE" , "X_gray_L_CLAHE","X_n"]

rslt={}
epochs=10
batch_size=16

score={}
matrix={}
report={}

for trans in transformation:
    
    with h5py.File(data_transorfmed_directory+trans+".h5", "r" ) as f :
        X=f["X"][:]
        y=f["y"][:]
    
    n,dx,dy,dz=X.shape
    model=create_model(dx, dy, dz)
         
    #model.summary()
          
    X_train, X_test, y_train, y_test= train_test_split(X,y,test_size=0.2)
    
    datagen = keras.preprocessing.image.ImageDataGenerator(featurewise_center=False,
                             featurewise_std_normalization=False,
                             width_shift_range=0.1,
                             height_shift_range=0.1,
                             zoom_range=0.2,
                             shear_range=0.1, 
                             rotation_range=10.)
    
    datagen.fit(X_train)
    
    model.compile(optimizer='adam',
              loss='categorical_crossentropy',
              metrics=['accuracy'])
    
    bestmodel_callback = tf.keras.callbacks.ModelCheckpoint(filepath=model_directory+"best_model_"+trans+".h5", 
                                                            verbose=0, 
                                                            monitor='accuracy', 
                                                            save_best_only=True)

    history = model.fit( datagen.flow(X_train, y_train, batch_size=16), 
                      epochs=epochs,
                      batch_size=batch_size,
                      verbose=1,
                      validation_data=(X_test, y_test),
                      callbacks=[bestmodel_callback])
    
    model=tf.keras.models.load_model(model_directory+"best_model_"+trans+".h5")
    
    evaluation=model.evaluate(X_test,y_test)
    score[trans]=evaluation
    
    y_sigmoid=model.predict(X_test)
    y_pred= np.argmax(y_sigmoid, axis=1)
    y_true=np.argmax(y_test,axis=1)
    matrix[trans]=confusion_matrix(y_true, y_pred)
    report[trans]=classification_report(y_true, y_pred, target_names=class_cancer)
    
    rslt[str(trans)]=history
    
## plot loss and accuracy for all models
plt.subplots(6,2, figsize=(15,15))
i=1
for key, value in rslt.items():
    print(key)
    
    plt.subplot(6,2, i)
    plt.plot(value.history["loss"] , label="Train loss")
    plt.plot(value.history["val_loss"] , label="Test loss")
    plt.legend()
    plt.title(key+' : Loss ')
    
    i=i+1
    
    plt.subplot(6,2, i)
    plt.plot(value.history["accuracy"] , label="Train accuracy")
    plt.plot(value.history["val_accuracy"] , label="Test accuracy")
    plt.legend()
    plt.title(key+' : Accuracy ')
    i=i+1
    
plt.show()  
    
## score, plot confuion matrix and classification report 
for key, value in score.items():
    print(f'{key}  ---- > loss : {value[0]:.{2}f}  // accuracy : {value[1]:.{2}f}')


for key, value in report.items():
    print (f'Classification report for {key} : /n /n {value} ')
    
    
for key , value in matrix.items():
    disp=ConfusionMatrixDisplay(confusion_matrix=value,
                                display_labels=class_cancer)
    disp.plot()
    plt.title("Confusion matrix for "+key)
    plt.show()
    
    
    
    
## prediction test
 
x= "C:/Users/pc/Nextcloud/Python/GITHUB/Computer_vision_CNN/data/images/ISIC_0024329.jpg"
img = np.asarray(Image.open(x).resize((32,32)))
img.shape
plt.imshow(img)




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




model_list=["rgb_h", "gray_L_HE", "gray_L_CLAHE", "gray_HE", "gray","n"]


rst={}
for model in model_list :
    
    im= image_transformation(img, model)
    im=im.reshape(-1,im.shape[0],im.shape[1],im.shape[2])    
    m = keras.models.load_model(model_directory+'best_model_X_'+model+'.h5')
    r=m.predict(im)
    rst[model]=np.argmax(r)

rst
