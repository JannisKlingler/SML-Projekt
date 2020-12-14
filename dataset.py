import matplotlib.pyplot as plt
import scipy as sp
import numpy as np
import tensorflow as tf
from tensorflow import keras


frames = 10
(x_train, y_train), (x_test, y_test) = keras.datasets.mnist.load_data()


x_train_rot = list(map(lambda b: list(map(lambda i: np.where(sp.ndimage.rotate(
    b, (i+1) * 360/frames, reshape=False) > 127.5, 1.0, 0.0).astype('float32'), range(frames))), x_train))

x_test_rot = list(map(lambda b: list(map(lambda i: np.where(sp.ndimage.rotate(
    b, (i+1) * 360/frames, reshape=False) > 127.5, 1.0, 0.0).astype('float32'), range(frames))), x_test))


for j in range(len(x_test_rot)):
    for i in np.random.choice(range(3, 10), 3, replace=False):
        x_test_rot[j][i] = np.zeros((28, 28))


train = np.transpose(np.array(x_train_rot), [0, 2, 3, 1])
test = np.transpose(np.array(x_test_rot), [0, 2, 3, 1])


np.save('C:/Users/Admin/Desktop/Python/Datasets/rotatingMNIST_train', train)
np.save('C:/Users/Admin/Desktop/Python/Datasets/rotatingMNIST_test', test)

#x_test = np.load('C:/Users/Admin/Desktop/Python/rotatingMNIST_test.npy')
