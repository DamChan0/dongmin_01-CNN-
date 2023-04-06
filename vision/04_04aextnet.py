############################################################################################################################################################
# 2023-04-04
############################################################################################################################################################
from keras.layers import Dense, Conv2D , MaxPooling2D, Flatten, Dropout, Activation, BatchNormalization
from keras.models import Sequential
from keras.callbacks import ModelCheckpoint
from keras import regularizers, optimizers
import numpy as np
import tensorflow as tf
import matplotlib.pyplot as plt
from matplotlib import pyplot
from keras.utils import to_categorical
from keras.datasets import cifar10
############################################################################################################################################################
# data loading
(x_train, y_train), (x_test, y_test) = cifar10.load_data()
xtrain = x_train.astype('float64')
xtest = x_test.astype('float64')

x_train, x_vaild = x_train[5000:], x_train[:5000]
y_train, y_vaild = y_train[5000:], y_train[:5000]

# print('x_train=',x_train.shape)
# print('y_train=',y_train.shape)
# print('x_test=',x_test.shape)

# data scaling
mean= np.mean(x_train, axis=(0,1,2,3))
std = np.std(x_train, axis=(0,1,2,3))
# print('mean=',mean)
# print('std=',std)
x_train = (x_train-mean)/(std+1e-7)
x_test = (x_test-mean)/(std+1e-7)
x_vaild = (x_vaild-mean)/(std+1e-7)
# print('x_train=',x_train)

from keras.preprocessing.image import ImageDataGenerator

# image generator
datagen = ImageDataGenerator(
        rotation_range= 15 ,
        width_shift_range= 0.1,
        height_shift_range= 0.1,
        horizontal_flip= True,
        vertical_flip= True,
)
datagen.fit(x_train)


# model
hidden_unit = 32
weight_decay = 1e-4
model = Sequential()
# conv1
model.add(Conv2D(hidden_unit, kernel_size= 3, padding= 'same', kernel_regularizer=regularizers.l2(weight_decay),
                 input_shape =x_train.shape[1:], activation= 'relu'))
model.add(BatchNormalization())
# conv2
model.add(Conv2D(hidden_unit, kernel_size= 3, padding='same', kernel_regularizer=regularizers.l2(weight_decay),
                 activation='relu'))
model.add(BatchNormalization())
# conv3
model.add(Conv2D(hidden_unit*2, kernel_size= 3, padding='same', kernel_regularizer=regularizers.l2(weight_decay),
                 activation='relu'))
model.add(BatchNormalization())
# conv4
model.add(Conv2D(hidden_unit*2, kernel_size= 3, padding='same', kernel_regularizer=regularizers.l2(weight_decay),
                 activation='relu'))
model.add(BatchNormalization())
# polling dropout
model.add(MaxPooling2D(pool_size=2))
model.add(Dropout(0.3))
# conv5
model.add(Conv2D(hidden_unit*4, kernel_size= 3, padding='same', kernel_regularizer=regularizers.l2(weight_decay),
                 activation='relu'))
model.add(BatchNormalization())
# conv6
model.add(Conv2D(hidden_unit*4, kernel_size= 3, padding='same', kernel_regularizer=regularizers.l2(weight_decay),
                 activation='relu'))
model.add(BatchNormalization())
# polling dropout
model.add(MaxPooling2D(pool_size=2))
model.add(Dropout(0.45))
# Flatten
model.add(Flatten())
model.add(Dense(10, activation='softmax'))
# summuary
model.summary()

# trainign hyperparameter >> batch size // epchos // optimizer
batch_size = 128
epochs = 120

checkout = ModelCheckpoint(filepath='model.100.h5', verbose= 1,
                           save_best_only=True)
# early_checkpoint = 

optimizer = optimizers.Adam(learning_rate=0.0001) 
model.compile(loss='categorical_crossentropy', optimizer=optimizer)
history = model.fit(datagen.flow(x_train,y_train, batch_size= batch_size), callbacks= [checkout],
                    steps_per_epoch= x_train.shape[0] ,epochs=epochs, verbose=2, validation_data=(x_vaild,y_vaild) )

_,train_acc = model.evaluate(x_train, y_train)
_,test_acc = model.evaluate(x_test, y_test)
# _ >> 리스트의 마지막을 할당
print('trian: %.3f , TEST : %.3f' %(train_acc, test_acc))

# training score
pyplot.plot(history.history['accuracy'], label='accuracy')
pyplot.plot(history.history['val_accuracy'], label='val_accuracy')
pyplot.legend()
pyplot.show()
