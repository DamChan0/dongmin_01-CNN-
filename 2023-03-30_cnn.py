############################################################################################################################################################
# CNN project
# 2023-03-30
############################################################################################################################################################
from keras.layers import Dense, Flatten, MaxPooling2D, Dropout, Conv2D
from keras import Sequential
import tensorflow as tf

model = Sequential()
model.add(Conv2D(filters=32 , kernel_size=(3,3), strides= 1, padding="same" , activation= tf.nn.relu, input_shape = (28,28,1)))
model.add(MaxPooling2D(pool_size= (2,2), strides= 2))
model.add(Conv2D(filters=64 , kernel_size=(3,3), strides= 1, padding="same" , activation= tf.nn.relu))
model.add(MaxPooling2D(pool_size= (2,2), strides= 2))

model.add(Flatten())
model.add(Dropout(0.3))
model.add(Dense(64, activation= "relu"))
model.add(Dropout(0.5))
model.add(Dense(10 , activation= "softmax"))

model.summary()