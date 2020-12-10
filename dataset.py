import matplotlib.pyplot as plt
import scipy as sp
import numpy as np
import tensorflow as tf
from tensorflow import keras


frames = 10
(x_train, y_train), (x_test, y_test) = keras.datasets.mnist.load_data()


# x_train = list(map(lambda b: list(map(lambda i: np.where(sp.ndimage.rotate(
#    b, (i+1) * 360/frames, reshape=False) > 127.5, 1.0, 0.0).astype('float32'), range(frames))), x_train))

x_test = list(map(lambda b: list(map(lambda i: np.where(sp.ndimage.rotate(
    b, (i+1) * 360/frames, reshape=False) > 127.5, 1.0, 0.0).astype('float32'), range(frames))), x_test[9000:10000]))


# for j in range(len(x_test)):
#    for i in np.random.randint(10, size=2):
#        x_test[j][i] = np.zeros((28, 28))

#train = np.array(x_train)
test = np.array(x_test)

#np.save('rotatingMNIST_train', train)
np.save('rotatingMNIST_test2', test)

#x_test = np.load('C:/Users/Admin/Desktop/Python/rotatingMNIST_test.npy')


plt.figure(figsize=(20, 20))
for j in range(10):
    for i in range(frames):
        ax = plt.subplot(10, frames, j*10+(i+1))
        plt.imshow(test[j][i], cmap='gray')
        ax.get_xaxis().set_visible(False)
        ax.get_yaxis().set_visible(False)
plt.show()
