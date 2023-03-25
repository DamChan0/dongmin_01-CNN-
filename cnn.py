############################################################################################################################################################
# make simple model 
# 2023-03-17
############################################################################################################################################################
import os
os.environ['TF_CPP_MIN_LOG_LEVEL'] = '2'
import tensorflow as tf
tf.keras.utils.set_random_seed(42)
tf.config.experimental.enable_op_determinism()
from tensorflow import keras
from keras.api._v2 import keras
from keras.utils import plot_model
from sklearn.model_selection import train_test_split

############################################################################################################################################################
class mycallback(tf.keras.callbacks.Callback):    
    def on_epoch_end(self, epoch, logs=None):
        if(logs.get('accuracy')>0.90):
            print('정확도가 90%에 도달했습니다')
            self.model.stop_training= True
mycallback1  = mycallback()

(train_input, train_target), (test_input, test_target) = \
    keras.datasets.fashion_mnist.load_data()

train_scaled = train_input.reshape(-1, 28, 28, 1) / 255.0

train_scaled, val_scaled, train_target, val_target = train_test_split(
    train_scaled, train_target, test_size=0.2, random_state=42)


model = keras.Sequential()

model.add(keras.layers.Conv2D(32, kernel_size=3, activation='relu', padding='same', input_shape=(28,28,1)))
model.add(keras.layers.MaxPooling2D(2))
model.add(keras.layers.Conv2D(64, kernel_size=3, activation='relu', padding='same'))
model.add(keras.layers.MaxPooling2D(2))
model.add(keras.layers.Flatten())
model.add(keras.layers.Dense(100, activation= 'relu' ))
model.add(keras.layers.Dropout(0.4))
model.add(keras.layers.Dense(10, activation='softmax'))
#  model layers set
#  model.summary()


model.compile(optimizer= 'adam', loss= 'sparse_categorical_crossentropy', metrics= 'accuracy')
callback = keras.callbacks.ModelCheckpoint('best-cnn-model.h5', save_best_only=True)
earlycallback  = keras.callbacks.EarlyStopping(patience= 2 , restore_best_weights=
                                               True)

history = model.fit(train_scaled, train_target, epochs=20, validation_data=(val_scaled, val_target)
                    , callbacks= [callback, earlycallback])


import matplotlib.pyplot as plt

# plt.plot(history.history['loss'])
# plt.plot(history.history['val_loss'])
# plt.xlabel('epoch')
# plt.ylabel('loss')
# plt.legend(['train', 'val'])
# plt.show()

model.evaluate(val_scaled, val_target)

predict1 = model.predict(val_scaled[2:3])       
#  차원 유지를 위한 슬라이싱

print(predict1)

classes = ['티셔츠', '바지', '스웨터', '드레스', '코트',
           '샌달', '셔츠', '스니커즈', '가방', '앵클 부츠']

import numpy as np
print( classes[np.argmax(predict1)])

model.layers



























