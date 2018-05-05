# -*- coding: utf-8 -*-
"""
Created on Sat Jul 22 15:04:39 2017

@author: manjunath.a
"""

from skimage.io import imread
import pandas as pd
from lxml import etree
from os import walk
import os
import numpy as np
from sklearn.model_selection import train_test_split

from keras.callbacks import EarlyStopping, ModelCheckpoint
from keras.models import Sequential
from keras.layers import Dense, Dropout, Activation, Flatten
from keras.layers import Conv2D, MaxPooling2D
from keras import backend as K
from keras.optimizers import RMSprop, Adam
from sklearn.metrics import accuracy_score, precision_score, recall_score, f1_score



def check_threshold(y_probability, y_actual, threshold, positive_class,
    negative_class):
    y_pred = [positive_class if y >= threshold else negative_class
     for y in y_probability]
    accuracy = accuracy_score(y_actual, y_pred)
    precision = precision_score(y_actual, y_pred)
    recall = recall_score(y_actual, y_pred)
    fscore = f1_score(y_actual, y_pred)
    #recall = precision_score(y_actual, y_pred)
    #precision = recall_score(y_actual, y_pred)
    return threshold, accuracy, precision, recall, fscore

def sweep_thresholds(y_probability, y_actual, start, stop, step,
    positive_class=1, negative_class=0):
    sweep = [ check_threshold(y_probability, y_actual, t,
     positive_class, negative_class) for t in np.arange(start, stop, step)]
    return np.array(sweep)
    
def get_best_fscore(sweep):
    max_fscore = np.max(sweep[:, 4])
    best_fscore = sweep[sweep[:, 4] == max_fscore]
    return best_fscore
  

### STEP 1
# Reading the input images and getting them into a pandas dataframe   

folder = '/home/img/task/JPEGImages'

files = []
for (dirpath, dirnames, filenames) in walk(folder):
    files.extend(filenames)  

imagepixelvalues_list, images_dict,fileids = [],{},[]    
for wavpath in files:
    inp_image = imread(os.path.join(folder, wavpath))
    #print(wavpath)
    images_dict[wavpath] = inp_image
    imagepixelvalues_list.append(inp_image)
    fileids.append(int(wavpath.split('.')[0]))


input_df = pd.DataFrame({'ImageId':fileids, 'pixelvalues':imagepixelvalues_list})

### STEP 2
# Parsing all xml files for output labels in images and getting the file, it’s labels data into dataframe
########### Looping

folder_ = '/home/img/task/Annotations'

files_ = []
for (dirpath, dirnames, filenames) in walk(folder_):
    files_.extend(filenames)  

labellist, labeldict, fileids_ = [],{},[]    
for wavpath in files_:
    xmlfile = (os.path.join(folder_, wavpath) )
    xmldoc = etree.parse(xmlfile)  
    imglist = []
    for name in xmldoc.xpath("//name"):
        imglist.append(name.text.strip())
        #print( name.text.strip() )
    labellist.append(imglist)
    #print(imglist)
    labeldict[wavpath] = imglist
    fileids_.append(int(wavpath.split('.')[0]))
 
           
label_df = pd.DataFrame({'ImageId':fileids_, 'Labels':labellist})

### STEP 3        
# Merge the image data dataframe and label details dataframe into a single dataframe
data = pd.merge(input_df, label_df, on='ImageId')

### STEP 4
# Input data and Output data Preparation
### Output label preparation                  
from sklearn.preprocessing import MultiLabelBinarizer

mlb = MultiLabelBinarizer()
Y = mlb.fit_transform(data.Labels)

### input data preparation
X=np.empty((len(data),720, 1280, 3))
for i in range(0,len(data)):
    X[i] = data.pixelvalues[i]


#Convert binary matrix back to labels
#y_predict_label = mlb.inverse_transform(Y) 


# train - test(Validatin) data split 
# I have used stratify=Y[:,2] in data split because this class label is highly skewed  

x_train , Xt, y_train, Yt = train_test_split(X,Y,test_size=0.4,random_state=100, stratify=Y[:,2])
x_val , x_test, y_val, y_test = train_test_split(Xt,Yt,test_size=0.5,random_state=100, stratify=Yt[:,2])

K.set_image_dim_ordering('tf')

x_train = x_train.astype('float32')
x_test  = x_test.astype('float32')    
x_val  = x_val.astype('float32')     
    
x_train /= 255
x_test /= 255
x_val /= 255
print('x_train shape:', x_train.shape)
print('x_test shape:', x_test.shape)
print('y_train shape:', y_train.shape)
print('y_test shape:', y_test.shape)

### STEP 5
# CNN Model building
early_stopping_epochs = 12

earlyStopping = EarlyStopping(monitor='val_loss', patience=early_stopping_epochs, verbose=1)


model = Sequential()
model.add(Conv2D(8, kernel_size=(3, 3), input_shape=(720, 1280, 3)))
model.add(Activation('relu'))

model.add(MaxPooling2D(pool_size=(2, 2)))
model.add(Dropout(0.3))

model.add(Conv2D(16, (4, 4)))
model.add(Activation('relu'))

model.add(MaxPooling2D(pool_size=(2, 2)))
model.add(Dropout(0.3))

model.add(Flatten()) 
model.add(Dense(256))
model.add(Activation('relu'))
model.add(Dropout(0.4))

model.add(Dense(128))
model.add(Activation('relu'))
model.add(Dropout(0.45))

model.add(Dense(48))
model.add(Activation('relu'))
model.add(Dropout(0.5))

model.add(Dense(8))
model.add(Activation('sigmoid'))


model.compile(loss='binary_crossentropy', optimizer="Adam", metrics=['accuracy'])

check = ModelCheckpoint(filepath="best_model_check_ctf-adam-6-2-2_verify.hdf5", monitor='val_loss', verbose=1, save_best_only = True)    

model.fit(x_train, y_train, batch_size=8, epochs=68,callbacks=[earlyStopping, check],validation_data=(x_test,y_test))


### STEP 6

############# Finding probability threshold that gives maximum Fscore for individual classes
# Model Prediction on validation data 
out = model.predict_proba(x_test)
out = np.array(out)

bestthresholds = np.zeros(out.shape[1])
for i in range(out.shape[1]):
    y_pred = np.array(out[:,i])
    out_ = sweep_thresholds(y_pred, y_test[:,i], 0.05, 0.95, 0.02)
    #print(out)
    best_thresholds = get_best_fscore(out_)
    #print("\n\n", best_thresholds[0,],"\n\n")
    bestthresholds[i] = best_thresholds[0,][0]
print( "best thresholds", bestthresholds)

y_pred = np.array([[1 if out[i,j]>=bestthresholds[j] else 0 for j in range(y_test.shape[1])] for i in range(len(y_test))])

fscore = f1_score(y_test, y_pred, average='samples')  # Gives Fscore for Multi label multiclass classification
print("FSCORE=",fscore)


### STEP 7
#Model Prediction on Unseen test data and Train data using the best thresholds of every class obtained in previous step


#Model Prediction on Unseen test data
print("\n UNSEEN Test Data Results\n")
out = model.predict_proba(x_val)
out = np.array(out)   

y_pred = np.array([[1 if out[i,j]>=bestthresholds[j] else 0 for j in range(y_val.shape[1])] for i in range(len(y_val))])

fscore = f1_score(y_val, y_pred, average='samples')   # Gives Fscore for Multi label multiclass classification
print("FSCORE=",fscore)

#Model Prediction on Train  data

print("\n Train Data Results\n")
out = model.predict_proba(x_train)
out = np.array(out)   

y_pred = np.array([[1 if out[i,j]>=bestthresholds[j] else 0 for j in range(y_train.shape[1])] for i in range(len(y_train))])

fscore = f1_score(y_train, y_pred, average='samples')   # Gives Fscore for Multi label multiclass classification
print("FSCORE=",fscore)


##### STEP 8
# Final Evaluation on Unseen TestData( and Results :-

folder = '/home/scripts/images/'

files = []
for (dirpath, dirnames, filenames) in walk(folder):
    files.extend(filenames)  

imagepixelvalues_list, images_dict,file_ids = [],{},[]    
for wavpath in files:
    inp_image = imread(os.path.join(folder, wavpath))
    #print(wavpath)
    images_dict[wavpath] = inp_image
    imagepixelvalues_list.append(inp_image)
    file_ids.append(int(wavpath.split('.')[0].split('-')[1]))

Test_df = pd.DataFrame({'ImageId':file_ids, 'pixelvalues':imagepixelvalues_list})
       
Xtest=np.empty((len(Test_df),720, 1280, 3))
for i in range(0,len(Test_df)):
    Xtest[i] = Test_df.pixelvalues[i]

Xtest = Xtest.astype('float32')   
Xtest /= 255


out = model.predict_proba(Xtest)
out = np.array(out)   


Out = np.array([[1 if out[i,j]>=bestthresholds[j] else 0 for j in range(out.shape[1])] for i in range(len(out))])

labels = list(mlb.classes_)

y_predict_label = mlb.inverse_transform(Out)

Result_df = pd.DataFrame({'ImageId':file_ids, 'Labels':y_predict_label})

Result_df.to_csv("Task_Results.csv",index=False)

Result_df = pd.DataFrame({'ImageId':file_ids, 'Labels':y_predict_label, 'Labelvalues':list(Out) })

Result_df.to_csv("Task_Results_1.csv",index=False)


### END

