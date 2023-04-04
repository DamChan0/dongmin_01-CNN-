############################################################################################################################################################
# make simple model 
# 2023-03-16 


############################################################################################################################################################

import os
os.environ['TF_CPP_MIN_LOG_LEVEL'] = '2'
import tensorflow as tf
tf.keras.utils.set_random_seed(42)
tf.config.experimental.enable_op_determinism()
from tensorflow import keras
from keras.api._v2 import keras
from sklearn.model_selection import train_test_split
import matplotlib.pyplot as plt
############################################################################################################################################################


(train_input, train_target), (test_input, test_target) = \
    keras.datasets.fashion_mnist.load_data()

train_scaled = train_input / 255.0

train_scaled, val_scaled, train_target, val_target = train_test_split(
    train_scaled, train_target, test_size=0.2, random_state=42)    
# datasets download


model = keras.models.Sequential()


def model_gen(a_layer=None):
    model = keras.Sequential()
    model.add(keras.layers.Flatten(input_shape=(28, 28)))
    model.add(keras.layers.Dense(100, activation='relu'))
    if a_layer:
        model.add(a_layer)
    model.add(keras.layers.Dropout(0.3))
    model.add(keras.layers.Dense(10, activation='softmax'))
    return model

model = model_gen()                     # model layers set
model.compile(optimizer='adam' , loss='sparse_categorical_crossentropy', metrics= 'accuracy')
model_cb = keras.callbacks.ModelCheckpoint('best-model.h5', save_best_only = True)
early_stopping_cb = keras.callbacks.EarlyStopping(patience = 2, restore_best_weights = True)
history = model.fit(train_scaled, train_target, epochs= 20 , verbose= 2,
                    validation_data= (val_scaled, val_target), callbacks=[model_cb, early_stopping_cb])     # verbose = 0  >>> training Process not printed
    

model.save_weights('model-weights.h5')
model.save('model.h5')

model = keras.models.load_model('best-model.h5')
model.evaluate(val_scaled,val_target)
# plt.plot(history.history['loss'])
# plt.plot(history.history['val_loss'])
# plt.xlabel('epoch')
# plt.ylabel('loss')
# plt.legend(['train', 'val'])
# plt.show()

    

