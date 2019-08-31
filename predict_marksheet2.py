
import numpy
import cv2
from keras.preprocessing.image import ImageDataGenerator
from keras.models import Sequential
from keras.layers import Convolution2D, MaxPooling2D
from keras.layers import Flatten, Dense,Dropout,Activation
import matplotlib.pyplot as plt

img_width, img_height = 28 , 28
# input image dimensions
img_rows, img_cols = 28, 28
# number of convolutional filters to use
nb_filters = 32
# size of pooling area for max pooling
pool_size = (2, 2)
# convolution kernel size
kernel_size = (3, 3)

batch_size = 128
nb_classes = 10
nb_epoch = 12

def create_model():
    model = Sequential()
    model.add(Convolution2D(nb_filters, kernel_size[0], kernel_size[1],
                        border_mode='valid',
                        input_shape=(28,28,1)))
    model.add(Activation('relu'))
    model.add(Convolution2D(nb_filters, kernel_size[0], kernel_size[1]))
    model.add(Activation('relu'))
    model.add(MaxPooling2D(pool_size=pool_size))
    model.add(Dropout(0.25))        #add dropout for better results

    model.add(Flatten())
    model.add(Dense(128))
    model.add(Activation('relu'))
    model.add(Dropout(0.5))         #add dropout for better results
    model.add(Dense(nb_classes))
    model.add(Activation('softmax'))

    return model




#color conversion
import cv2
import numpy as np
from PIL import Image
import matplotlib.pyplot as plt
i=0

#function for predicting output from cropped images
def prediction(t):
 #reading image

 p1="marksheet.jpg"
 image=Image.open(p1)
 image=image.crop((t[0],t[1],t[2],t[3]))

 p2="conversion_1.jpg"
 image.save(p2)

 #with the help of otsu binarization and blackhat 
 image = cv2.imread(p2,1)

 gray = cv2.cvtColor(image,cv2.COLOR_BGR2GRAY)


 #Rectangular kernel with size 5x5
 kernel = cv2.getStructuringElement(cv2.MORPH_RECT,(2,2))

 #apply blackhat and otsu thresholding
 blackhat = cv2.morphologyEx(gray,cv2.MORPH_BLACKHAT,kernel)

 _,thresh = cv2.threshold(blackhat,0,255,cv2.THRESH_BINARY|cv2.THRESH_OTSU)
 plt.imshow(thresh)
 thresh = cv2.dilate(thresh,kernel)   #dilate thresholded image for better segmentation
 plt.imshow(thresh)



 thresh=cv2.resize(thresh,(28,28))
 p3="conversion_2.jpg"
 cv2.imwrite(p3,thresh)





 model = create_model()
 model.load_weights('mnistneuralnet17.h5')

 arr = numpy.array(thresh)
 arr=arr.reshape((img_width,img_height,1))
 arr = numpy.expand_dims(arr, axis=0)
 prediction = model.predict(arr)[0]
 bestclass = ''
 bestconf = -1
 for n in [0,1,2,3,4,5,6,7,8,9]:
  if(prediction[n] > bestconf):
   bestclass = str(n)
   bestconf = prediction[n]
 print ('I think this digit is a ' + bestclass + ' with ' + str(bestconf * 100) + '% confidence.')
 
 sheet1.write(t[4],t[5],bestclass)
 
 
 


#For saving into excel file
from xlwt import Workbook
wb=Workbook()
sheet1=wb.add_sheet("Sheet 1") #for adding sheets


#passing each cropped image of first horizontal part to prediction function
#prediction((64,570,100,597,0,0))
#prediction((130,570,170,597,0,1))
#prediction((201,572,231,595,0,2))#wrong before
#prediction((271,570,295,594,0,3))#wrong before
#prediction((330,570,381,597,0,4))
#prediction((400,570,441,597,0,5))
#prediction((472,570,511,597,0,6))
#prediction((541,571,570,601,0,7))#wrong berfore
#prediction((601,570,649,597,0,8))
#prediction((680,574,710,599,0,9))#wrong prediction

#passing each cropped image of second horizontal part to prediction function
#prediction((120,760,169,780,1,0))
#prediction((261,760,289,780,1,1))
#prediction((392,760,440,788,1,2))
prediction((534,761,570,785,1,3))# wrong before
#passing each cropped image of third horizontal part to prediction function
#prediction((129,800,159,821,2,0))
#prediction((260,800,290,821,2,1))
#prediction((399,800,431,827,2,2))
#prediction((540,800,568,820,2,3))
#passing each cropped image of fourth horizontal part to prediction function
#prediction((249,833,300,861,3,0))
#prediction((120,830,170,861,3,1))#wrong
#prediction((390,835,439,865,3,2))#wrong
#prediction((531,841,579,863,3,3))

wb.save("marksheet.xls")






