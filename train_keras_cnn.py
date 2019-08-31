
from __future__ import print_function
import numpy as np
np.random.seed(1337)  # for reproducibility

from keras.models import Sequential
from keras.layers import Dense, Dropout, Activation, Flatten
from keras.layers import Convolution2D, MaxPooling2D
from keras import backend as K
from utils.dataset import load_dataset,encode
import pandas as pd
from sklearn.preprocessing  import LabelEncoder , OneHotEncoder

batch_size = 128
nb_classes = 10
nb_epoch = 12

# input image dimensions
img_rows, img_cols = 28, 28
# number of convolutional filters to use
nb_filters = 32
# size of pooling area for max pooling
pool_size = (2, 2)
# convolution kernel size
kernel_size = (3, 3)

#load training data
train=pd.read_csv("dataset/mnist_train.csv")
X_train=train.iloc[:,1:]
Y_train=train.iloc[:,0]
Y_train=np.array(Y_train).reshape(60000,1)

encoder_ytrain=LabelEncoder()
Y_train[:,0]=encoder_ytrain.fit_transform(Y_train[:,0])
onehotencoder=OneHotEncoder(categorical_features= [0])
Y_train=onehotencoder.fit_transform(Y_train).toarray()


#load testing data
test=pd.read_csv("dataset/mnist_test.csv")
X_test=test.iloc[:,1:]
Y_test=test.iloc[:,0]
Y_test=np.array(Y_test).reshape(10000,1)

encoder_ytest=LabelEncoder()
Y_test[:,0]=encoder_ytest.fit_transform(Y_test[:,0])
onehotencoder=OneHotEncoder(categorical_features= [0])
Y_test=onehotencoder.fit_transform(Y_test).toarray()


#convert to float
Y_train = Y_train.astype("float32")
Y_test = Y_test.astype("float32")

#refactoring train and test sets


#normalize data
X_train /= 255
X_test /= 255
X_train=np.array(X_train)
X_test=np.array(X_test)

#handling compatibility issues with image shape in theano and tensoflow backend
if K.image_dim_ordering() == 'th':
    X_train = X_train.reshape(X_train.shape[0], 1, img_rows, img_cols)
    X_test = X_test.reshape(X_test.shape[0], 1, img_rows, img_cols)
    input_shape = (1, img_rows, img_cols)
else:
    X_train = X_train.reshape(X_train.shape[0], img_rows, img_cols, 1)
    X_test = X_test.reshape(X_test.shape[0], img_rows, img_cols, 1)
    input_shape = (img_rows, img_cols, 1)

#create model
model = Sequential()
model.add(Convolution2D(nb_filters, kernel_size[0], kernel_size[1],
                        border_mode='valid',
                        input_shape=input_shape))
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

model.compile(loss='categorical_crossentropy',
              optimizer='adadelta',
              metrics=['accuracy'])

model.fit(X_train, Y_train, batch_size=batch_size, nb_epoch=nb_epoch,
          verbose=1, validation_data=(X_test, Y_test))
score = model.evaluate(X_test, Y_test, verbose=0)
print('Test cost:', score[0])
print('Test accuracy:', score[1])

print("[INFO] saving model to disk...")

#save model to disk
model.save_weights('mnistneuralnet17.h5')