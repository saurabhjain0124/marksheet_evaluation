# marksheet_evaluation
use cnn with keras for recognizing marks in marksheet

## objective
This project is based on recognizing marks on marksheet using Deep Learning techniques.Using convolution neural nets on the handwritten images and after applying different image processing techniques hand written digits are recognized and are stored in excel files. 

code for training of model is in train_keras_cnn.py

code for recognizing marks from marksheet is in predict_marksheet2.py

pretrained model is in mnistneuralnet17.h5

excel file for storing result is marksheet.xls

## outcome
In this project,CNN model is employed to recognize 19 cropped images of marks correctly out of 22 cropped images of marksfrom the scanned copy of the marksheet.

## dependencies
* keras
* tensorflow
* numpy
* pandas
* pillow
* opencv
* matplotlib

## steps
Following steps are followed for recognizing marks from student marksheet: 

1)Prepare modelor training CNN model using MNIST. 

2)Import scanned copy of original mark sheet. 

3)Image segmentation or cropping of marksheet for retrieving the needed parts. 

4)Pre-processing of cropped part. 

5)Digitsrecognition on cropped part. 

## DATASET COLLECTION 
MNIST DATASET-   MNIST dataset is used in training. 
MNIST (Mixed National Institute of Standards and Technology) database is a dataset for handwritten digits, distributed by Yann Lecun’s THE MNIST DATABASE of handwritten digits website. 

link:http://yann.lecun.com/exdb/mnist/

## complete overview of project
 ### Model preparation
 model is prepared by using keras which is very Popular deep learning library. Architecture of my model are as follow.
 
 1)Convolution layer: size 3*3, activation relu, filters 32 
 
 2)Convolution layer: size 3*3, activation relu, filters 64 
 
 3)Max pool layer: size 2*2 
 
 4)Dropout: 25% 
 
 5)Dense layer: size 128 neurons, activation relu 
 
 6)Dropout: 50% 
 
 7)Fully connected layer: size 10 neurons, activation SoftMax. 

 ### Importing original mark sheet 
Import original with the help of open Function of pillow library. 

 ### Cropping 
Since in mark sheet we have to recognize marks which are present in specific location of mark sheet so at first, we have to crop that part by using famous pillow library. 

 ### Pre-processing 
 It also has these following parts- 

 a. Importing image: read cropped image with the help of famous OpenCV library. 

 b. Convert it into grey scale: convert the cropped image to grayscale image as grayscale images are always best suited for image           processing tasks. The gray scaled image is shown below. 
          
 c. Applying Otsu binarization and black hat on it: create a rectangular kernel of size 5*5 and apply Blackhat morphological operation       on the grayscale image. Blackhat is a morphological operation which reveals dark regions on a light background. Since our task is to     detect handwritten digits written on a paper, we need to reveal the details written by pen (dark regions) ignoring the paper (light     background) we use Blackhat operation here. And the resulting image can be seen below. 

d.  Dilation: dilate image in order to increase width of image. 

e.  Finally resizing it [28,28]:finally resize it to size 28*28 because trained model on MNIST dataset accept images of size [28,28]. 

 
###  Recognition 
recognize final resized image by using the pretrained CNN model.

### further extensions
* first is to generalize this code beacuse this code is valid for presicting marks on this marksheet only.
* second is how to avoid cropping manually.
* third is to predict for multidigits.



 

 
