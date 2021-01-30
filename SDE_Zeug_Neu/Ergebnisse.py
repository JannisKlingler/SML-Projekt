#import datasets as data
import time
import matplotlib.pyplot as plt
from mpl_toolkits.axes_grid1 import ImageGrid
import numpy as np
import tensorflow as tf
import SDE_Tools
import AE_Tools

frames = 50

data_path = 'C:/Users/bende/Documents/Uni/Datasets/'

Z_org = np.load(data_path+'Results_SDE_Ball_Z_org_{}frames.npy'.format(frames))
X_org = np.load(data_path+'Results_SDE_Ball_X_org_{}frames.npy'.format(frames))

Z_enc = np.load(data_path+'Results_SDE_Ball_Z_enc_{}frames.npy'.format(frames))
Z_rec = np.load(data_path+'Results_SDE_Ball_Z_rec_{}frames.npy'.format(frames))
X_rec = np.load(data_path+'Results_SDE_Ball_X_rec_{}frames.npy'.format(frames))

print('shape:',X_rec.shape)

A = 11
B = 10

fig, axs = plt.subplots(A, B)

xl = np.linspace(0,frames,frames)
for i in range(8):
    axs[0,i].plot(xl, -Z_enc_List[i,:frames,0])
    axs[0,i].xaxis.set_visible(False)
    axs[0,i].yaxis.set_visible(False)
    for j in range(10):
        axs[1+j,i].imshow(X_org[i,j,:,:], cmap='gray')
        axs[1+j,i].axis('off')

for i in range(A):
    axs[i,8].axis('off')

axs[0,9].plot(xl, -Z_rec_List[i,:frames,0])
axs[0,9].xaxis.set_visible(False)
axs[0,9].yaxis.set_visible(False)
for j in range(10):
    axs[1+j,9].imshow(X_rec_List[7,j,:,:,0], cmap='gray')
    axs[1+j,9].axis('off')

plt.show()
