import tensorflow as tf
from keras.models import load_model
from keras.preprocessing import image
import matplotlib.pyplot as plt
import numpy as np
import os
# 
# 
# 
import cv2 as cv



def load_image(img_path, show=False):

    img = tf.keras.utils.load_img(img_path, target_size=(64, 64))
    img_tensor = tf.keras.utils.img_to_array(img)                    # (height, width, channels)
    img_tensor = np.expand_dims(img_tensor, axis=0)         # (1, height, width, channels), add a dimension because the model expects this shape: (batch_size, height, width, channels)
    img_tensor /= 255.                                      # imshow expects values in the range [0, 1]

    return img_tensor

    # load model
model = load_model("Fatih\\CN_and_NN.h5")

    # image path
img_path = 'Fatih\\Fatih\\115.jpg'   
# dosyaYolu="Dataset/deneme2"
# files=os.listdir(dosyaYolu)
#for f in files:
    # load a single image
new_image = load_image(img_path)
print(new_image)

        # check prediction
pred = model.predict(new_image)
print(pred)

if pred[0][0] >= 0.5:
    prediction = 'normal'
else:
    prediction = 'cataract'
print(prediction)
