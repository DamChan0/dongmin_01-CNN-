from keras.models import Sequential
from keras.layers import Dense, Conv2D, MaxPooling2D,AveragePooling2D,  Flatten, Dropout, BatchNormalization
from keras.callbacks import ModelCheckpoint, LearningRateScheduler
import numpy as np
from keras.utils import np_utils
from keras.datasets import mnist

rows =28
cols = 28
num_classes = 10

    # use Keras to import pre-shuffled MNIST database
(X_train, y_train), (X_test, y_test) = mnist.load_data()
X_train, X_test = X_train[10000:], X_test[:10000]
y_train, y_test = y_train[10000:], y_train[:10000]
# normalize the data to accelerate learning
mean = np.mean(X_train)
std = np.std(X_train)
X_train = (X_train-mean)/(std+1e-7)
X_test = (X_test-mean)/(std+1e-7)

################################# when image reshape too many values error is occur################################
# X_train = np.reshape(X_train, (X_train.shape[0], rows, cols, 1))
# X_test = np.reshape(X_test, (X_test.shape[0], rows, cols, 1))
####################################################################################################################
input_shape = (rows, cols, 1)

y_train = np_utils.to_categorical(y_train, num_classes)
y_test = np_utils.to_categorical(y_test, num_classes)

import matplotlib.pyplot as plt
# matplotlib inline
import matplotlib.cm as cm
import numpy as np

# plot first six training images
fig = plt.figure(figsize=(20,20))
for i in range(12):
    ax = fig.add_subplot(2, 6, i+1, xticks=[], yticks=[])
    ax.imshow(X_train[i], cmap='gray')
    ax.set_title(str(y_train[i]))
plt.show()
def visualize_input(img, ax):
    ax.imshow(img, cmap='gray')
    width, height = img.shape
    thresh = img.max()/2.5
    for x in range(width):
        for y in range(height):
            ax.annotate(str(round(img[x][y],2)), xy=(y,x),
                        horizontalalignment='center',
                        verticalalignment='center',
                        color='white' if img[x][y]<thresh else 'black')

fig = plt.figure(figsize = (12,12)) 
ax = fig.add_subplot(111)
visualize_input(X_train[0], ax)
print("The MNIST database has a training set of %d examples." % len(X_train))
print("The MNIST database has a test set of %d examples." % len(X_test))


# model = Sequential()

# # C = conv2d / S = Maxpooling2d or average pooling2d
# # C1
# model.add(Conv2D(filters = 6, kernel_size= 5, strides = 1, padding = 'same', activation='tanh',
#                  input_shape = (28,28,1)))
# # S2
# model.add(AveragePooling2D(pool_size = (2,2), padding='valid'))
# # C3
# model.add(Conv2D(filters = 16, kernel_size= 5, strides=1 ,activation='tanh', padding= 'valid'))
# # S4
# model.add(AveragePooling2D(pool_size = (2,2),strides=2, padding = 'valid'))
# # c5
# model.add(Conv2D(filters = 120, kernel_size=2, strides=2, padding='valid', activation='tanh'))
# model.add(Flatten())
# # Fully connected layer
# model.add(Dense(84, activation='tanh'))
# model.add(Dense(10, activation='softmax'))  

# # model.summary()

# def Ir_gradient(epoch):
#     if epoch <=2:
#         Ir = 5e-4
#     elif epoch >2 and epoch <=5:
#         Ir = 2e-4
#     elif epoch >5 and epoch <=9:
#         Ir = 5e-5
#     else:
#         Ir = 1e-5
        
#     return Ir

# Ir_scheduler = LearningRateScheduler(Ir_gradient)
# checkout = ModelCheckpoint(filepath = 'LeNet_model.h5', monitor = 'val-acc', verbose =1,
#                            save_best_only= True)
# callbacks = [checkout, Ir_scheduler]

# model.compile(loss = 'categorical_crossentropy', optimizer = 'sgd', metrics = ['accuracy'])

# hist = model.fit(X_train, y_train, epochs = 20, batch_size= 32,
#                  validation_data=(X_test, y_test), callbacks = callbacks,
#                  verbose=1, shuffle= True)
