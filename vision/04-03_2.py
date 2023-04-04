############################################################################################################################################################
# 2023-04-03
############################################################################################################################################################
from keras.layers import Dense, Conv2D , MaxPooling2D, Flatten, Dropout
from keras.models import Sequential
import numpy as np
import matplotlib.pyplot as plt
from matplotlib import pyplot
from sklearn.datasets import make_blobs
from keras.utils import to_categorical
############################################################################################################################################################
# which model?
# how many layers
# what activation?
# what optimizer?
# what Regulatoryization?
# data split train:test = 8:2 or 7:3 add vaild >> 6:2:2, 70:15:15

x,y = make_blobs(n_samples= 1000 , centers=3, n_features= 2,
                 cluster_std= 2, random_state=2)

y = to_categorical(y)

n_tarin = 800
train_X, test_X = x[:n_tarin, :], x[n_tarin:, :]
train_y, test_y = y[:n_tarin], y[n_tarin:]

model = Sequential()
model.add(Dense(25, input_dim=2, activation='relu'))
model.add(Dense(3, activation='softmax'))

model.compile(loss='categorical_crossentropy', optimizer='adam', metrics=['accuracy'])
model.summary()

history = model.fit(train_X, train_y, validation_data=(test_X, test_y), epochs=1000, verbose=1)

_,train_acc = model.evaluate(train_X, train_y)
_,test_acc = model.evaluate(test_X, test_y)
# _ >> 리스트의 마지막을 할당
print('trian: %.3f , TEST : %.3f' %(train_acc, test_acc))

# training score
pyplot.plot(history.history['accuracy'], label='accuracy')
pyplot.plot(history.history['val_accuracy'], label='val_accuracy')
pyplot.legend()
pyplot.show()
