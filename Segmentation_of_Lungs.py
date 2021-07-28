#Import Libraries

import pandas as pd
import numpy as np
import os
import cv2
from skimage.io import imread,imshow
import time
from skimage.transform import resize
import matplotlib.pyplot as plt
import keras as K
import tensorflow as tf
from PIL import Image, ImageOps
from keras.preprocessing.image import ImageDataGenerator
tf.random.set_seed(7)

masks_list = os.listdir('../input/chest-xray-masks-and-labels/Lung Segmentation/masks')

masks_list_final = []

for string in masks_list:
    new_string = string.replace("_mask", "")
    masks_list_final.append(new_string)

img_list = os.listdir('../input/chest-xray-masks-and-labels/Lung Segmentation/CXR_png')

img_list_final = list(set(img_list) & set(masks_list_final))

img_list_final = list(set(img_list) & set(masks_list_final))

x_train = np.zeros((len(img_list_final), 128,128,3), dtype=np.uint8)

y_train = np.zeros((len(img_list_final), 128,128), dtype=np.uint8)

start = time.time()
for i in range(len(img_list_final)):
    img = cv2.imread('../input/chest-xray-masks-and-labels/Lung Segmentation/CXR_png/'+str(img_list_final[i]),cv2.IMREAD_COLOR)
    img=cv2.resize(img,(128,128))
    x_train[i]=img
    if i%50==0:
        print(i)
end = time.time()
print("Time Taken: ",end-start)

x_train = x_train / 255

imshow(x_train[5])
plt.show

begin = time.time()
for i in range(len(img_list_final)):
    img = cv2.imread('../input/chest-xray-masks-and-labels/Lung Segmentation/masks/'+str(masks_list[i]),cv2.IMREAD_GRAYSCALE)
    img = cv2.resize(img,(128,128))
    y_train[i]=img
    if i%50==0:
        print(i)
finish = time.time()
print("Time Taken: ",finish-begin)

y_train = y_train / 255

imshow(y_train[5])
plt.show

print(x_train.shape)
print(y_train.shape)

IMG_WIDTH = 128
IMG_HEIGHT = 128
IMG_CHANNELS = 3

inputs = tf.keras.layers.Input((IMG_WIDTH, IMG_HEIGHT, IMG_CHANNELS))


# s = tf.keras.layers.Lambda(lambda x: x/255)(inputs)

def dice_coef(y_true, y_pred):
    y_true_f = tf.keras.layers.Flatten()(y_true)
    y_pred_f = tf.keras.layers.Flatten()(y_pred)
    intersection = tf.keras.backend.sum(y_true_f * y_pred_f)
    return (2. * intersection + 1) / (tf.keras.backend.sum(y_true_f) + tf.keras.backend.sum(y_pred_f) + 1)


def dice_coef_loss(y_true, y_pred):
    return -dice_coef(y_true, y_pred)


# Contraction Path
c1 = tf.keras.layers.Conv2D(16, (3, 3), activation='relu', kernel_initializer='he_normal', padding='same')(inputs)
c1 = tf.keras.layers.Dropout(0.1)(c1)
c1 = tf.keras.layers.Conv2D(16, (3, 3), activation='relu', kernel_initializer='he_normal', padding='same')(c1)
p1 = tf.keras.layers.MaxPooling2D((2, 2))(c1)

c2 = tf.keras.layers.Conv2D(32, (3, 3), activation='relu', kernel_initializer='he_normal', padding='same')(p1)
c2 = tf.keras.layers.Dropout(0.1)(c2)
c2 = tf.keras.layers.Conv2D(32, (3, 3), activation='relu', kernel_initializer='he_normal', padding='same')(c2)
p2 = tf.keras.layers.MaxPooling2D((2, 2))(c2)

c3 = tf.keras.layers.Conv2D(64, (3, 3), activation='relu', kernel_initializer='he_normal', padding='same')(p2)
c3 = tf.keras.layers.Dropout(0.2)(c3)
c3 = tf.keras.layers.Conv2D(64, (3, 3), activation='relu', kernel_initializer='he_normal', padding='same')(c3)
p3 = tf.keras.layers.MaxPooling2D((2, 2))(c3)

c4 = tf.keras.layers.Conv2D(128, (3, 3), activation='relu', kernel_initializer='he_normal', padding='same')(p3)
c4 = tf.keras.layers.Dropout(0.2)(c4)
c4 = tf.keras.layers.Conv2D(128, (3, 3), activation='relu', kernel_initializer='he_normal', padding='same')(c4)
p4 = tf.keras.layers.MaxPooling2D(pool_size=(2, 2))(c4)

c5 = tf.keras.layers.Conv2D(256, (3, 3), activation='relu', kernel_initializer='he_normal', padding='same')(p4)
c5 = tf.keras.layers.Dropout(0.3)(c5)
c5 = tf.keras.layers.Conv2D(256, (3, 3), activation='relu', kernel_initializer='he_normal', padding='same')(c5)

# Expansive path
u6 = tf.keras.layers.Conv2DTranspose(128, (2, 2), strides=(2, 2), padding='same')(c5)
u6 = tf.keras.layers.concatenate([u6, c4])
c6 = tf.keras.layers.Conv2D(128, (3, 3), activation='relu', kernel_initializer='he_normal', padding='same')(u6)
c6 = tf.keras.layers.Dropout(0.2)(c6)
c6 = tf.keras.layers.Conv2D(128, (3, 3), activation='relu', kernel_initializer='he_normal', padding='same')(c6)

u7 = tf.keras.layers.Conv2DTranspose(64, (2, 2), strides=(2, 2), padding='same')(c6)
u7 = tf.keras.layers.concatenate([u7, c3])
c7 = tf.keras.layers.Conv2D(64, (3, 3), activation='relu', kernel_initializer='he_normal', padding='same')(u7)
c7 = tf.keras.layers.Dropout(0.2)(c7)
c7 = tf.keras.layers.Conv2D(64, (3, 3), activation='relu', kernel_initializer='he_normal', padding='same')(c7)

u8 = tf.keras.layers.Conv2DTranspose(32, (2, 2), strides=(2, 2), padding='same')(c7)
u8 = tf.keras.layers.concatenate([u8, c2])
c8 = tf.keras.layers.Conv2D(32, (3, 3), activation='relu', kernel_initializer='he_normal', padding='same')(u8)
c8 = tf.keras.layers.Dropout(0.1)(c8)
c8 = tf.keras.layers.Conv2D(32, (3, 3), activation='relu', kernel_initializer='he_normal', padding='same')(c8)

u9 = tf.keras.layers.Conv2DTranspose(16, (2, 2), strides=(2, 2), padding='same')(c8)
u9 = tf.keras.layers.concatenate([u9, c1], axis=3)
c9 = tf.keras.layers.Conv2D(16, (3, 3), activation='relu', kernel_initializer='he_normal', padding='same')(u9)
c9 = tf.keras.layers.Dropout(0.1)(c9)
c9 = tf.keras.layers.Conv2D(16, (3, 3), activation='relu', kernel_initializer='he_normal', padding='same')(c9)

outputs = tf.keras.layers.Conv2D(1, (1, 1), activation='sigmoid')(c9)

model = tf.keras.Model(inputs=[inputs], outputs=[outputs])
model.compile(optimizer='adam', loss="binary_crossentropy", metrics=[dice_coef, 'binary_accuracy'])
model.summary()

earlystopping = tf.keras.callbacks.EarlyStopping(monitor='loss', patience=10)
history = model.fit(x = x_train,
                    y = y_train,
                    epochs = 30,
                    batch_size = 16,
                   callbacks = [earlystopping])

test_list = os.listdir('../input/chest-xray-masks-and-labels/Lung Segmentation/test')

x_test = np.zeros((len(test_list), 128,128,3), dtype=np.uint8)
y_pred = np.zeros((len(test_list), 128,128,3), dtype=np.uint8)

start = time.time()
for i in range(len(test_list)):
    #img =  tf.keras.preprocessing.image.load_img(x_path_list[i], target_size=(128,128), color_mode='grayscale', interpolation='lanczos')
    img = cv2.imread('../input/chest-xray-masks-and-labels/Lung Segmentation/test/'+str(test_list[i]),cv2.IMREAD_COLOR)
    img=cv2.resize(img,(128,128))
    x_test[i]=img
    if i%50==0:
        print(i)
end = time.time()
print("Time Taken: ",end-start)

x_test = x_test / 255

x_test[5].shape

y_pred = model.predict(np.asarray(x_test))

imshow(x_test[90])
plt.show

imshow(y_pred[90])
plt.show

plt.figure(figsize=(20,20))
num = 0
for i in range(9):
    plt.subplot(3, 3, i + 1)
    plt.imshow(x_test[num],cmap='gray')
    num = num+9
    plt.axis('off')

plt.figure(figsize=(20,20))
num = 0
for i in range(9):
    plt.subplot(3, 3, i + 1)
    plt.imshow(y_pred[num],cmap='gray')
    num = num+9
    plt.axis('off')

