import matplotlib.pyplot as plt
import scipy as sp
import numpy as np
import tensorflow as tf
from tensorflow import keras


frames = 20
(x_train, y_train), _ = keras.datasets.mnist.load_data()


def rot(pic, frames):
    video = []
    for i in range(frames):
        im = sp.ndimage.rotate(pic, i*360/frames, reshape=False)
        video.append([im])
    return video


def dataset(size, frames):
    data = []
    for i in range(size):
        vid = rot(x_train[i], frames)
        data.append([vid])
    return data


data = dataset(1, 10)
print(len(data))
plt.figure(figsize=(20, 4))
plt.imshow(data[0, 1, 0, :, :])
plt.show()
