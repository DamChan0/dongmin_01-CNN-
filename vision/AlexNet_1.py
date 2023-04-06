################# this code is "Received assistance from Chat GPT"(input sentence is "make AelxNet code") ################################################
import keras
from keras.models import Sequential
from keras.regularizers import L2
from keras.layers import Dense, Dropout, Flatten, Conv2D, MaxPooling2D, AveragePooling2D, BatchNormalization
# fucking gpt damn
##################################################################################################################################
# Define the model
model = Sequential()

# Add the layers
model.add(Conv2D(96, kernel_size=(11, 11), strides=(4, 4), padding='valid', activation='relu', input_shape=(227, 227, 3)))
model.add(MaxPooling2D(pool_size=(3, 3), strides=(2, 2)))
model.add(BatchNormalization())
model.add(Conv2D(256, kernel_size=(5, 5), activation='relu'))
model.add(MaxPooling2D(pool_size=(3, 3), strides=(2, 2)))
model.add(BatchNormalization())
model.add(Conv2D(384, kernel_size=(3, 3),strides=1, padding='same', activation='relu'
                 , kernel_regularizer=L2(0.0005)))
model.add(BatchNormalization())
model.add(Conv2D(384, kernel_size=(3, 3), activation='relu', kernel_regularizer=L2(0.0005)))
model.add(BatchNormalization())
model.add(Conv2D(256, kernel_size=(3, 3), activation='relu', kernel_regularizer=L2(0.0005)))
model.add(BatchNormalization())
model.add(MaxPooling2D(pool_size=(3, 3), strides=(2, 2), padding='valid'))

model.add(Flatten())
model.add(Dense(4096, activation='relu'))
model.add(Dropout(0.5))
model.add(Dense(4096, activation='relu'))
model.add(Dropout(0.5))
model.add(Dense(1000, activation='softmax'))

model.summary()

# Compile the model
model.compile(loss=keras.losses.categorical_crossentropy, 
              optimizer=keras.optimizers.SGD(lr=0.01, momentum=0.9),
              metrics=['accuracy'])

# model.fit(x_train, y_train, batch_size=128, epochs=90,
#           validation_data=(X_test, y_test), verbose=1, callback = [reduce_Ir])




