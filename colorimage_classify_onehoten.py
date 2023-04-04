############################################################################################################################################################
# CNN project
# 2023-03-30
############################################################################################################################################################
from keras.layers import Dense, Flatten, MaxPooling2D, Dropout, Conv2D
from keras.callbacks import ModelCheckpoint
from keras import Sequential
import tensorflow as tf
import keras
from keras.datasets import cifar10
import numpy as np
import matplotlib.pyplot as plt
from keras.utils import np_utils
############################################################################################################################################################

(x_train, y_train), (x_test, y_test) = cifar10.load_data()

fig = plt.figure(figsize=(20,5))

for i in range(36):
    ax = fig.add_subplot(3, 12, i + 1, xticks=[], yticks=[])
    ax.imshow(np.squeeze(x_train[i]))
    
x_train = x_train.astype('float32') / 225.0
x_test = x_test.astype('float32') / 225.0

# one hot encoding
num_classes = len(np.unique(y_train))
y_train = keras.utils.to_categorical(y_train, num_classes)
y_test = keras.utils.to_categorical(y_test, num_classes)

# data spilt

(x_train, x_vaild) = x_train[5000:], x_test[:5000]
(y_train, y_vaild) = y_train[5000:], y_test[:5000]

# print('x_train.shape:', x_train.shape, 'y_train.shape:', y_train.shape)
# print('훈련데이터수' , x_train.shape[0])
# print('테스트 데이터수' , x_test.shape[0])
# print('검증 데이터수' , x_vaild.shape[0])
# print(num_classes)

# model setting Alexnet

model= Sequential()
model.add(Conv2D(filters=16 , kernel_size=(3,3), activation='relu', input_shape=(32,32,3)))
model.add(MaxPooling2D(pool_size=(2,2)))

model.add(Conv2D(filters=32 , kernel_size=(3,3), activation='relu', input_shape=(32,32,3)))
model.add(MaxPooling2D(pool_size=(2,2)))

model.add(Conv2D(filters=64 , kernel_size=(3,3), activation='relu', input_shape=(32,32,3)))
model.add(MaxPooling2D(pool_size=(2,2)))

model.add(Flatten())
model.add(Dense(500, activation='relu'))
model.add(Dropout(0.4))
model.add(Dense(num_classes, activation='softmax'))

model.compile(loss='categorical_crossentropy', optimizer='adam', metrics=['accuracy'])

# model training
class mycallback(tf.keras.callbacks.Callback):    
    def on_epoch_end(self, epoch, logs=None):
        if(logs.get('accuracy')>0.80):
            print('정확도가 80%에 도달했습니다')
            self.model.stop_training= True
mycallback1  = mycallback()
callback = keras.callbacks.ModelCheckpoint('model.weight.best.h5', save_best_only=True)

# model.fit(x_train, y_train, batch_size=32, epochs = 100, validation_data=(x_vaild, y_vaild),
#                      callbacks= [mycallback1, callback] ,
#                      verbose= 2, shuffle= True)


model.load_weights('model.weight.best.h5')

score = model.evaluate(x_test, y_test, verbose= 0)
print('\n', 'Test loss:', score[0], 'Test accuracy:', score[1])