# %tensorflow_version 1.x
import time

import cv2
import os
import numpy as np

from tensorflow.lite.python import interpreter as interpreter_wrapper

model_path = 'model_best_quantized.tflite'

interpreter = interpreter_wrapper.Interpreter(model_path=model_path)
interpreter.allocate_tensors()

input_details = interpreter.get_input_details()
output_details = interpreter.get_output_details()

print(input_details)
print(output_details)

while 1:

  imgPath = input("input image: ")

  if imgPath == "":
    break

  img = cv2.imread(imgPath)
  img = cv2.resize(img, (224,224))

  img = img[np.newaxis, ...]

  start_time = time.time()
  interpreter.set_tensor(input_details[0]['index'], img)
  interpreter.invoke()

  output_data0 = interpreter.get_tensor(output_details[0]['index'])
  output_data0 = output_data0.astype(np.float)
  print(output_data0/255)

  print("--- %s seconds ---" % (time.time() - start_time))
