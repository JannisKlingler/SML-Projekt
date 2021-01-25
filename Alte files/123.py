import imageio
import numpy as np
from mpl_toolkits.axes_grid1 import ImageGrid
import matplotlib.pyplot as plt

import numpy as np
from keras.models import Sequential
from keras.layers import AveragePooling2D

datapath = 'C:/Users/Admin/Desktop/AVI/'
savepath = 'C:/Users/Admin/Desktop/Python/Datasets/'
frames = 2*8

# %%
a = 'person'
box = '_boxing_d'
hand = '_handclapping_d'
wave = '_handwaving_d'
b = '_uncomp.avi'

mean_pool = AveragePooling2D(pool_size=2, strides=2)
model = Sequential([mean_pool])

x_train = []
for i in range(25):
    for j in [box, hand, wave]:
        for k in range(4):
            filename = datapath+a+str(i+1).zfill(2)+j+str(k+1)+b
            print(filename)
            vid = imageio.get_reader(filename,  'ffmpeg')
            array = []
            for l, image in enumerate(vid.iter_data()):
                array.append(image[:, 20:140, 0] / 255)
            array = np.array(array)
            array = array[:np.shape(array)[0]-np.shape(array)[0] % frames]
            array = array.reshape(np.shape(array)[0], 120, 120, 1)
            array = model.predict(array)
            array = np.squeeze(array)
            for m in range(int(np.shape(array)[0]/frames)):
                video_seq = array[m*frames:m*frames+frames]
                x_train.append(array[[0, 2, 4, 6, 8, 10, 12, 14]])
                x_train.append(array[[1, 3, 5, 7, 9, 11, 13, 15]])
x_train = np.array(x_train)
x_train = np.transpose(x_train, (0, 2, 3, 1))

np.save(savepath + 'video_meanpool', x_train)
print(np.shape(x_train))

# %%
'''
x_train = np.load(savepath + 'video_meanpool_train_x_noflow155_89.npy')
x_rec = np.load(savepath + 'video_meanpool_x_rec_noflow155_89.npy')
frames = 8
# print(np.shape(x_train))
#6,24,64,73
#43,73,92,104,31,69
a=[]
for j, i in enumerate([43,73,92,104,36,69]):
    a.append(np.concatenate((x_train[i,:,:,0:4],x_rec[i,:,:,4:8]),axis=2))
a=np.array(a)
#print(np.shape(a[0]))

fig = plt.figure(figsize=(15, 15), dpi=200)
#index = range(65,70)#np.random.randint(100, size=5)
grid = ImageGrid(fig, 111,  nrows_ncols=(2*3, frames), axes_pad=[0.1,0.1],)
plot = np.zeros((6*frames,60, 60))
for j in range(6):
    plot[j*frames:(j+1)*frames] = np.transpose(a[j],(2,0,1))#np.concatenate((x_rec[i,:,:,0:4],x_rec[i,:,:,4:8]),axis=2)
    #plot[(j*2+1)*frames:(j*2+2)*frames] = np.transpose(x_rec[i], (2, 0, 1))
#plot2 = []
#for i in range(6):
#    plot2.append(plot[:,:,i:i+frames])

#print(np.shape(np.array(plot2)))

#plot=np.reshape(plot,(48,60,60))
#plot.tanspose((2, 0, 1))
for ax, im in zip(grid, plot):
    plt.gray()
    ax.get_xaxis().set_visible(False)
    ax.get_yaxis().set_visible(False)
    ax.imshow(im)

plt.savefig('C:/Users/Admin/Desktop/Ergebnisse/movies.png')
plt.show()

#%%

#print(np.shape(x_train[0,:,:,0:4]))
#print(np.shape(x_rec[0,:,:,0:4]))
a=[]
for j, i in enumerate([43,73,92,104,37,69]):
    a.append(np.concatenate((x_train[i,:,:,0:4],x_rec[i,:,:,0:4]),axis=2))
a=np.array(a)
print(np.shape(a))
#print(np.shape(np.concatenate((x_train[0,:,:,0:4],x_rec[0,:,:,0:4]),axis=2)))
'''
