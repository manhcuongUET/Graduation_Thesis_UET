# predict model
from cv2 import cv2
import glob
from keras.models import load_model
import numpy as np
import time
import tensorflow as tf

from keras.backend.tensorflow_backend import set_session
config = tf.ConfigProto()
config.gpu_options.per_process_gpu_memory_fraction = 0.7
set_session(tf.Session(config=config))

model = load_model('model_best_weights.h5')
# model.summary()

while 1:

  imgPath = input("input image: ")

  if imgPath == "":
    break

  img = cv2.imread(imgPath)
  img = cv2.resize(img, (32,32))
  img = img.astype(np.float32)
  img = img/255
  # print(img)

  img = img[np.newaxis, ...]

  start_time = time.time()
  result = model.predict(img)
  result = np.squeeze(result)
  print(result)
  result = (round(result[0], 2),round(result[1], 2),round(result[2], 2))
  print("--- %s seconds ---" % (time.time() - start_time))
  print(result)


