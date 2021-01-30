from mpl_toolkits.axes_grid1 import ImageGrid
import numpy as np
import scipy as sp
import matplotlib.pyplot as plt
from matplotlib import animation
from scipy.optimize import fsolve
from skimage import draw
import tensorflow as tf
import time
from math import pi, log, sin, cos, exp
# %%
import SDE_Tools


Ntrain = 10000 # 16 GB Arbeitsspeicher funktioniert, aber ALLE Programme davor schließen!
Ntest = 10000
frames = 50
simulated_frames = 200
T = 50
n = 1
#data_path = 'C:/Users/Admin/Desktop/Python/Datasets/'
data_path = 'C:/Users/bende/Documents/Uni/SML-Projekt/Ball/'


X_0 = np.array([np.zeros(Ntrain), np.ones(Ntrain)])
X_0 = np.transpose(X_0, [1, 0])

# mu : R^d -> R^d
def mu(x):
    m = np.array([x[1], -(3*2*pi/T)**2*x[0]])
    #m = np.array([x[1],1])
    return m

# sigma: R^d -> R^(nxd)
def sigma(x):
    s = np.array([[0.1], [0.2]])
    #s = np.zeros((d,n))
    return s


x_train = np.array(list(map(lambda i: SDE_Tools.ItoDiffusion(
    2, n, T, frames, simulated_frames, X_0[i], mu, sigma), range(Ntrain))))
x_test = np.array(list(map(lambda i: SDE_Tools.ItoDiffusion(
    2, n, T, frames, simulated_frames, X_0[i], mu, sigma), range(Ntest))))
x_train = x_train[:Ntrain, :, :-1]
x_test = x_test[:Ntest, :, :-1]
print('x_train shape:', x_train.shape)


List = []
for i in range(Ntrain):
    list = []
    for j in range(frames):
        #position = [min(max(x_train[i, j, 0], -2), 2)/4+0.5, 0.5]
        position = [x_train[i, j, 0]/16+0.5, 0.5]
        radius = 0.12
        arr = np.zeros((28, 28))
        ro, co = draw.disk((position[0] * 28, position[1] *
                            28), radius=radius*28, shape=arr.shape)
        arr[ro, co] = 1
        list.append(arr)
    List.append(list)
x_train_pictures = np.array(List)
print('x_train_pictures shape:', x_train_pictures.shape)

List = []
for i in range(Ntest):
    list = []
    for j in range(frames):
        #position = [min(max(x_train[i, j, 0], -2), 2)/4+0.5, 0.5]
        position = [x_test[i, j, 0]/8+0.5, 0.5]
        radius = 0.12
        arr = np.zeros((28, 28))
        ro, co = draw.disk((position[0] * 28, position[1] *
                            28), radius=radius*28, shape=arr.shape)
        arr[ro, co] = 1
        list.append(arr)
    List.append(list)
x_test_pictures = np.array(List)
print('x_test_pictures shape:', x_test_pictures.shape)

np.save(data_path+'SDE_Ball_train', x_train_pictures)
np.save(data_path+'SDE_Ball_test', x_test_pictures)

fig, axs = plt.subplots(8, 20)
for i in range(8):
    for j in range(20):
        axs[i, j].imshow(x_train_pictures[i, j, :, :])
        axs[i, j].axis('off')
plt.show()
