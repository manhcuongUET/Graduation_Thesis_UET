# Tensorflow 2x
import tensorflow as tf

print(tf.__version__)
from tensorflow.keras import backend, models, layers, optimizers
import numpy as np
from tensorflow.keras.layers import GlobalAveragePooling2D
from tensorflow.keras.callbacks import EarlyStopping
from sklearn.model_selection import train_test_split
from tensorflow.keras.utils import plot_model
from IPython.display import display
from PIL import Image
from tensorflow.keras.preprocessing.image import ImageDataGenerator
import os, shutil
from tensorflow.keras.models import Model

np.random.seed(42)

# Specify the base directory where images are located.
base_dir = '/content/drive/MyDrive/Khoá luận/Data xoai full)'
# Specify the traning, validation, and test dirrectories.
train_dir = os.path.join(base_dir, 'Train')
test_dir = os.path.join(base_dir, 'Validation')
# Normalize the pixels in the train data images, resize and augment the data.
train_datagen = ImageDataGenerator(
    rescale=1. / 255,  # The image augmentaion function in Keras
    shear_range=0.2,
    zoom_range=0.2,  # Zoom in on image by 20%
    horizontal_flip=True)  # Flip image horizontally
# Normalize the test data imagees, resize them but don't augment them
test_datagen = ImageDataGenerator(rescale=1. / 255)
train_generator = train_datagen.flow_from_directory(
    train_dir,
    target_size=(224, 224),
    batch_size=16,
    class_mode='categorical')
test_generator = test_datagen.flow_from_directory(
    test_dir,
    target_size=(224, 224),
    batch_size=16,
    class_mode='categorical')

# Load InceptionV3 library
from tensorflow.keras.applications.inception_v3 import InceptionV3

# Always clear the backend before training a model
backend.clear_session()
# InceptionV3 model and use the weights from imagenet
conv_base = InceptionV3(weights='imagenet',  # Useing the inception_v3 CNN that was trained on ImageNet data.
                        include_top=False)

# Connect the InceptionV3 output to the fully connected layers
InceptionV3_model = conv_base.output
pool = GlobalAveragePooling2D()(InceptionV3_model)
dense_1 = layers.Dense(512, activation='relu')(pool)
output = layers.Dense(3, activation='softmax')(dense_1)

# Create an example of the Archictecture to plot on a graph
model_example = models.Model(inputs=conv_base.input, outputs=output)
# plot graph
plot_model(model_example)

# Define/Create the model for training
model_InceptionV3 = models.Model(inputs=conv_base.input, outputs=output)
# Compile the model with categorical crossentropy for the loss function and SGD for the optimizer with the learning
# rate at 1e-4 and momentum at 0.9
model_InceptionV3.compile(loss='categorical_crossentropy',
                          optimizer=optimizers.SGD(lr=1e-4, momentum=0.9),
                          metrics=['accuracy'])

# Import from tensorflow the module to read the GPU device and then print
from tensorflow.python.client import device_lib

print(device_lib.list_local_devices())

# Execute the model with fit_generator within the while loop utilizing the discovered GPU
import tensorflow as tf

with tf.device("/device:GPU:0"):
    history = model_InceptionV3.fit_generator(
        train_generator,
        epochs=10,
        validation_data=test_generator,
        verbose=1,
        callbacks=[EarlyStopping(monitor='val_accuracy', patience=5, restore_best_weights=True)])
