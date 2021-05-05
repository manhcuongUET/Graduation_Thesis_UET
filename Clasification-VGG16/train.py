# %tensorflow_version 1.x

import tensorflow
print(tensorflow.__version__)
from keras import applications, optimizers
from keras.preprocessing.image import ImageDataGenerator
from keras.models import Sequential, Model
from keras.layers import Dropout, Flatten, Dense, Input
import keras
from keras.callbacks import EarlyStopping, ModelCheckpoint
import keras.backend as K
import os
import tensorflow as tf
import matplotlib
import matplotlib.pyplot as plt
import numpy as np
from tensorflow.keras.datasets import cifar10
from sklearn.metrics import classification_report

from keras.backend.tensorflow_backend import set_session
config = tf.ConfigProto()
config.gpu_options.per_process_gpu_memory_fraction = 0.6
set_session(tf.Session(config=config))

print(keras.__version__)

print("***************************************device***************************************")
if tf.test.gpu_device_name():
   print('Default GPU Device: {}'.format(tf.test.gpu_device_name()))
else:
   print("Please install GPU version of TF")
print("***************************************device***************************************")

# SET ALL THE PARAMETERS
# weights_path = 'models/vgg16.h5'
img_width, img_height = 64, 64
train_data_dir = "./Train"
validation_data_dir = "./Validation"
epochs = 20
batch_size = 16

classes = 3

# LOAD VGG16
input_tensor = Input(shape=(img_width,img_height,3))
Final_model = applications.VGG16(weights='imagenet',
                           include_top=False,
                           input_tensor=input_tensor)

# CREATE A TOP MODEL
top_model = Flatten()(Final_model.output)
top_model = Dense(256, activation='relu')(top_model)
top_model = Dropout(0.5)(top_model)
top_model = Dense(classes, activation='softmax')(top_model)

new_model = Model(input=Final_model.input, output=top_model)

# LOCK THE TOP CONV LAYERS
for layer in new_model.layers[:15]:
    layer.trainable = False

new_model.summary()

# COMPILE THE MODEL
new_model.compile(loss='categorical_crossentropy',
                  optimizer='sgd',
                  metrics=['accuracy'])

# CREATE THE IMAGE GENERATORS
train_datagen = ImageDataGenerator(rescale=1./255)
validation_datagen = ImageDataGenerator(rescale=1./255)

train_generator = train_datagen.flow_from_directory(
                                                    train_data_dir,
                                                    target_size=(img_height,img_width),
                                                    batch_size=batch_size,
                                                    class_mode='categorical',
                                                    shuffle=True)

validation_generator = validation_datagen.flow_from_directory(
                            validation_data_dir,
                            target_size=(img_height,img_width),
                            batch_size=batch_size,
                            class_mode='categorical')


STEP_SIZE_TRAIN=train_generator.n//train_generator.batch_size
STEP_SIZE_VALID=validation_generator.n//validation_generator.batch_size

early_stop = EarlyStopping(monitor='loss', min_delta=0.001, patience=3, mode='min', verbose=1)
checkpoint = ModelCheckpoint('v1-64px-2-05_model_best_weights.h5', monitor='loss', verbose=1, save_best_only=True, mode='min', period=1)

#  FIT THE MODEL
H = new_model.fit_generator(
    train_generator,
    steps_per_epoch=STEP_SIZE_TRAIN,
    epochs=epochs,
    validation_data=validation_generator,
    validation_steps=STEP_SIZE_VALID,
    callbacks = [early_stop,checkpoint])
