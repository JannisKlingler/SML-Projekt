import matplotlib.pyplot as plt
import numpy as np
import tensorflow as tf
import tensorflow_probability as tfp
import scipy as sp
from keras import backend as K
import SDE_Tools
import AE_Tools
tfd = tfp.distributions
# import datasets as data

'''
# Needed for gpu support on some machines
config = tf.compat.v1.ConfigProto(
    gpu_options=tf.compat.v1.GPUOptions(per_process_gpu_memory_fraction=0.8))
config.gpu_options.allow_growth = True
session = tf.compat.v1.Session(config=config)
tf.compat.v1.keras.backend.set_session(session)
'''

########################################################
# %% hyperparameter
epochs = 10
latent_dim = 5  # Dimensionality for latent variables. 20-30 works fine.
batch_size = 10  # ≥100 as suggested by Kingma in Autoencoding Variational Bayes.
train_size = 5000  # Data points in train set. Choose accordingly to dataset size.
test_size = 10 # Data points in test set. Choose accordingly to dataset size.
batches = int(train_size / batch_size)
frames = 10  # Number of images in every datapoint. Choose accordingly to dataset size.
armortized_len = 3  # Sequence size seen by velocity encoder network.
act = 'relu'  # Activation function 'tanh' is used in odenet.
T = 1  # number of seconds of the video
fps = T/frames
n = 1
pictureWidth = 28
pictureHeight = 28
pictureColors = 1
M = 3
CNN_complexity = 15
SDE_Net_complexity = 20


ode_integration = 'trivialsum'  # options: 'DormandPrince' , 'trivialsum'
dt = 0.1  # Step size for numerical integration. Increasing dt reduces training time
# but may impact predictive qualitiy. 'trivialsum'  uses dt = 1 and only reconstruction
# loss as training criteria. This is very fast and works surprisingly well.
data_path = 'C:/Users/Admin/Documents/Uni/SML-Projekt/'
job = 'rotatingMNIST'  # Dataset for training. Options: 'rotatingMNIST' , 'bouncingBalls'

# %%
########################################################
# Datensatz laden oder erstellen
try:
    #raise Exception('Ich will den Datensatz neu erstellen')
    x_train = np.load('C:/Users/bende/Documents/Uni/SML-Projekt/rotatingMNIST_train.npy')
    x_test = np.load('C:/Users/bende/Documents/Uni/SML-Projekt/rotatingMNIST_test.npy')
    x_train = x_train[0:train_size]
    x_test = x_test[0:test_size]
except:
    print('Dataset is being generated. This may take a few minutes.')
    (x_train, y_train), (x_test, y_test) = tf.keras.datasets.mnist.load_data()
    x_train = x_train[0:train_size]
    x_test = x_test[0:test_size]
    x_train_rot = list(map(lambda b: list(map(lambda i: np.where(sp.ndimage.rotate(
        b, (i+1) * 360/frames, reshape=False) > 127.5, 1.0, 0.0).astype('float32'), range(frames))), x_train))
    x_test_rot = list(map(lambda b: list(map(lambda i: np.where(sp.ndimage.rotate(
        b, (i+1) * 360/frames, reshape=False) > 127.5, 1.0, 0.0).astype('float32'), range(frames))), x_test))
    for j in range(len(x_test_rot)):
        for i in np.random.choice(range(3, 10), 3, replace=False):
            x_test_rot[j][i] = np.zeros((28, 28))
    x_train = np.transpose(np.array(x_train_rot), [0, 2, 3, 1])
    x_test = np.transpose(np.array(x_test_rot), [0, 2, 3, 1])
    try:
        np.save('C:/Users/bende/Documents/Uni/SML-Projekt/rotatingMNIST_train', x_train)
        np.save('C:/Users/bende/Documents/Uni/SML-Projekt/rotatingMNIST_test', x_test)
    except:
        print('could not save Dataset')
    print('Dataset generated')


# Dim: train_size x frames x pictureWidth x pictureHeight x pictureColors
x_train = np.transpose(np.array([x_train]), (1, 4, 2, 3, 0))
print('train-shape:', x_train.shape)


# Dim: test_size x frames x pictureWidth x pictureHeight x pictureColors
x_test = np.transpose(np.array([x_test]), (1, 4, 2, 3, 0))


########################################################
# Datensatz für Encoder erstellen

#Dim: train_size*(frames-M+1) x pictureWidth x pictureHeight x (M*pictureColors)
x_train_longlist = AE_Tools.make_training_data(x_train, train_size, frames, M)
print('new_train_shape:',x_train_longlist.shape)


########################################################
# SDE_VAE definieren

class SDE_Autoencoder(tf.keras.Model):
    def __init__(self, encoder, reconstructor, decoder):
        super(SDE_Autoencoder, self).__init__()
        self.encoder = encoder
        self.decoder = decoder
        self.reconstructor = reconstructor

    def fullcall(self, inputs):
        print('SDE_AE input:',inputs.shape)
        frames_List = tf.split(inputs, len(inputs[0]), axis=1)
        Z_enc_List = []
        for x in frames_List:
            Z_enc = self.encoder(x[:,0,:,:,:])
            Z_enc_List.append(Z_enc)
            print('Z_enc:',Z_enc.shape, inputs.shape)
            #Z_enc = tf.split(Z_enc, batch_size, axis=0)
            #Z_enc = tf.stack(Z_enc, axis=0)
        Z_enc_List = tf.stack(Z_enc_List,axis=1)
        print('Z_enc_List:',Z_enc_List.shape)
        Z_0 = Z_enc_List[:,0,:]
        Z_rec_List = self.reconstructor(Z_0)
        print('Z_rec_List:',Z_rec_List.shape)
        X_rec_List = []
        for i in range(frames-M+1):
            X_rec_List.append(self.decoder(Z_rec_List[:,i,:]))
        X_rec_List = tf.stack(X_rec_List, axis=1)
        print('X_rec_List:',X_rec_List.shape)
        return [Z_enc_List, Z_rec_List, X_rec_List]

    def call(self, inputs):
        Z_enc_List, Z_rec_List, X_rec_List = self.fullcall(inputs)
        return X_rec_List


########################################################
# Model aufbauen und trainieren

P_enc = AE_Tools.LocalEncoder(latent_dim, M, pictureWidth, pictureHeight, pictureColors, act, complexity=CNN_complexity)
P_dec = AE_Tools.FramewiseDecoder(latent_dim, pictureWidth, pictureHeight, pictureColors, act, complexity=CNN_complexity)

ms_Net = SDE_Tools.mu_sig_Net(latent_dim, n, act, SDE_Net_complexity)
reconstructor = SDE_Tools.make_Tensorwise_Reconstructor(latent_dim*pictureColors, n, T, frames-M+1, ms_Net, batch_size)
SDE_AE = SDE_Autoencoder(P_enc, reconstructor, P_dec)

rec_loss = AE_Tools.make_binary_crossentropy_rec_loss(M)
p_loss = SDE_Tools.make_pointwise_Loss(T, frames-M+1, ms_Net, SDE_AE, getOwnInputs=True)
cv_loss = SDE_Tools.make_covariance_Loss(T, frames-M+1, batch_size, ms_Net, SDE_AE, getOwnInputs=True)
ss_loss = SDE_Tools.make_sigma_size_Loss(ms_Net, SDE_AE, getOwnInputs=True)

#TODO: SDE_AE gibt jetzt mehr aus. Loss anpassen
loss = lambda x_org,outp: 10*rec_loss(x_org, outp) + 1*p_loss(x_org) + 5*cv_loss(x_org) #+ 0*ss_loss(x_org)


SDE_AE.compile(optimizer='adam', loss=loss, metrics=[lambda x,out: rec_loss(x,out)])
SDE_AE.fit(x_train_longlist, x_train_longlist, epochs=epochs, batch_size=batch_size, shuffle=False)


'''
AE = AE_Tools.SimpleAutoencoder(P_enc, P_dec)
AE.compile(optimizer='adam', loss=rec_loss)
AE.fit(x_train_longlist, x_train_longlist[:,:,:,1], epochs=epochs, batch_size=batch_size*(frames-M+1), shuffle=False)
'''




######################################

#x_test = data.create_dataset(dataset_size=100, frames=10, picture_size=28, object_number=3)
k = 0
x_test_org = x_train[0:batch_size]
print('x_test_org:',x_test_org.shape)
rec_imgs = SDE_AE(x_train_longlist[0:batch_size,:,:,:,:])
print('rec_imgs:',rec_imgs.shape)
enc_lat = list(map(lambda i: P_enc(x_train_longlist[i,:,:,:,:]), range(batch_size)))
enc_lat = tf.stack(enc_lat, axis=0)
print('enc_lat:',enc_lat.shape)
rec_lat = reconstructor(enc_lat[:,0,:])
print('rec_lat:',rec_lat.shape)

fig, axs = plt.subplots(9, 10)
for i in range(2):
    for j in range(10):
        axs[4*i, j].imshow(x_test_org[i,j,:,:,0], cmap='gray')
        if j > 0 and j < 9:
            axs[4*i+1, j].plot(np.linspace(1,latent_dim,latent_dim),enc_lat[i,j-1,:],'o')
            axs[4*i+2, j].plot(np.linspace(1,latent_dim,latent_dim),rec_lat[i,j-1,:],'o')
            axs[4*i+3, j].imshow(rec_imgs[i,j-1,:,:,0], cmap='gray')
for i in range(5):
    axs[8, 2*i].plot(np.linspace(1,frames-M+1,frames-M+1),enc_lat[i,:,:],'-')
    axs[8, 2*i+1].plot(np.linspace(1,frames-M+1,frames-M+1),rec_lat[i,:,:],'-')
plt.show()
