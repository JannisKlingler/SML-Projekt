#import datasets as data
import time
import matplotlib.pyplot as plt
from mpl_toolkits.axes_grid1 import ImageGrid
import numpy as np
import tensorflow as tf
import SDE_Tools
import AE_Tools

frames = 20

data_path = 'C:/Users/bende/Documents/Uni/Datasets/'

X_org = np.load(data_path+'Results_SDE_rotMNIST_X_org_{}frames.npy'.format(frames))

Z_enc = np.load(data_path+'Results_SDE_rotMNIST_Z_enc_{}frames.npy'.format(frames))
Z_rec = np.load(data_path+'Results_SDE_rotMNIST_Z_rec_{}frames.npy'.format(frames))
X_rec = np.load(data_path+'Results_SDE_rotMNIST_X_rec_{}frames.npy'.format(frames))

print('shape:',X_rec.shape)

'''
fig, axs = plt.subplots(10, 19)
k=0
for i in range(5):
    for j in range(19):
        axs[2*i,j].imshow(X_org[i+k,j,:,:,0], cmap='gray')
        axs[2*i,j].axis('off')
        axs[2*i+1,j].imshow(X_rec[i+k,j,:,:,0], cmap='gray')
        axs[2*i+1,j].axis('off')
'''

fig, axs = plt.subplots(2, 5)

xl = np.linspace(0,frames-1,frames-1)
for i in range(5):
    axs[0,i].plot(xl, Z_enc[i,:frames-1,:3])
    axs[1,i].plot(xl, Z_rec[i,:frames-1,:3])


plt.show()
