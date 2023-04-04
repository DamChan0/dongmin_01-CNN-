############################################################################################################################################################
# make simple model 
# 2023-03-17
############################################################################################################################################################
import tensorflow as tf
import os
os.environ['TF_CPP_MIN_LOG_LEVEL']='2'
tf.keras.utils.set_random_seed(42)
tf.config.experimental.enable_op_determinism()
from tensorflow import keras
from keras.api import keras
import matplotlib.pyplot as plt
############################################################################################################################################################

model = keras.models.load_model('best-cnn-model.h5')
model.layers

############################################################################################################################################################
#  가중치 시각화
# conv = model.layers[0]

# print(conv.weights[0].shape, conv.weights[1].shape)

# conv_weights = conv.weights[0].numpy()

# # print(conv_weights)
# print(conv_weights.mean(), conv_weights.std())pip ins

# plt.hist(conv_weights.reshape(-1, 1))
# plt.xlabel('weight')
# plt.ylabel('count')
# plt.show()


# fig, axs = plt.subplots(2, 16, figsize=(15, 2))

# for i in range(2):
#     for j in range(16):
#         axs[i,j].imshow(conv_weights[:,:,0,i*16+j], vmin=-0.5, vmax=0.5) 
#         axs[i,j].axis('off')
        
# plt.show()                  

############################################################################################################################################################
no_training_model = keras.Sequential()
no_training_model.add(keras.layers.Conv2D(64, (3, 3), activation='relu', padding='same', input_shape=(28,28,1)))

no_training_conv = no_training_model.layers[0]

no_training_conv_weights = no_training_conv.weights[0].numpy()
# print(no_training_conv_weights)

# plt.hist(no_training_conv_weights.reshape(-1, 1))
# plt.xlabel('weight')
# plt.ylabel('count')
# plt.show()

# 함수형 api

input = keras.Input(shape=(784,))
dense1 = keras.layers.Dense(100, activation='sigmoid')
dense2 = keras.layers.Dense(10,activation='softmax')
hidden = dense1(input)
output = dense2(hidden)

# print(model.input)

conv_actti = keras.Model(model.input, model.layers[0].output)

conv1 = keras.Model(no_training_model.input, no_training_model.layers[0].output)

# dataset_download = tf.keras.datasets.mnist

(train_input, train_target), (test_input, test_target) = tf.keras.datasets.fashion_mnist.load_data()

# fig, axs = plt.subplots(5, 6, figsize=(6,5))
# for i in range(5):
#     for j in range(6):
#         axs[i,j].imshow(train_input[i*5+j],cmap='gray')  
#         axs[i,j].axis('off')
        
# plt.imshow(train_input[0], cmap='gray')

plt.show()

input  = train_input[0].reshape(-1, 28, 28, 1) / 225.0
feature_map = conv_actti.predict(input)
print(feature_map.shape)

fig, axs = plt.subplots(4, 8, figsize=(15,8))

for i in range(4):
    for j in range(8):
        axs[i,j].imshow(feature_map[0,:,:,i*8+j])  
        axs[i,j].axis('off')

plt.show()

