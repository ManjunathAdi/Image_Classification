# Image_Multilabelmulticlass_Classification

Solved using keras (tensorflow as backend)

Used CNN (Convolutional Neural Network) to solve the classification problem.

Steps followed to solve the Task : -

1.] Reading the input images and getting them into a pandas dataframe
2.] Parsing all xml files for output labels in images and getting the file, it’s labels data into dataframe
3.] Merge the image data dataframe and label details dataframe into a single dataframe
4.] Input data(train, validation, test) and Output Data Preparation
For train-validation- test data split, I have used stratify= label that is highly skewed and to ensure that we have enough images of this label in all train, validation, test data splits.  
5.] CNN Model building, hyper parameter tuning 
6.] Model Prediction on validation data & Finding probability threshold that gives maximum Fscore for individual classes
7.] Model Prediction on Unseen test data and Train data using the best thresholds of every class obtained in previous step
8.] Final Evaluation on Unseen TestData and Results 

The architecture of the model –
Convolution layer using 8 filters of size=(3, 3) on input data of shape=(720, 1280, 3)  # Activation = 'relu' 
MaxPooling of pool_size=(2, 2)      with Dropout = 0.3
Convolution layer using 16 filters of size=(4, 4)   # Activation = 'relu'
MaxPooling of pool_size=(2, 2)       with Dropout  = 0.3
Flatten the input
Neural network layer with hidden neurons = 256        ### Activation = 'relu' with Dropout = 0.4
Neural network layer with hidden neurons = 128        ### Activation = 'relu' with Dropout = 0.45
Neural network layer with hidden neurons = 48          ### Activation = 'relu' with Dropout = 0.5
Neural network layer with output neurons = (8)           ### Activation = 'sigmoid'

COST Function: loss='binary_crossentropy',           &           OPTIMIZER Used: optimizer="Adam"
