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
epochs = 5
latent_dim = 6  # Dimensionality for latent variables. 20-30 works fine.
batch_size = 50  # ≥100 as suggested by Kingma in Autoencoding Variational Bayes.
train_size = 5000  # Data points in train set. Choose accordingly to dataset size.
test_size = 10 # Data points in test set. Choose accordingly to dataset size.
batches = int(train_size / batch_size)
frames = 20  # Number of images in every datapoint. Choose accordingly to dataset size.
armortized_len = 3  # Sequence size seen by velocity encoder network.
act = 'relu'  # Activation function 'tanh' is used in odenet.
act_ms_Net = 'tanh'
T = 50  # number of seconds of the video
fps = T/frames
n = 1
pictureWidth = 28
pictureHeight = 28
pictureColors = 1
M = 2
CNN_complexity = 20
SDE_Net_complexity = 50
forceHigherOrder = False


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

#Dim: train_size, (frames-M+1) x pictureWidth x pictureHeight x (M*pictureColors)
#x_train_longlist = AE_Tools.make_training_data(x_train, train_size, frames, M)
#print('new_train_shape:',x_train_longlist.shape)


########################################################
# Definitionen

derivatives = SDE_Tools.make_tensorwise_derivatives(M, frames, fps)


encoder = AE_Tools.make_Clemens_encoder(latent_dim)
decoder = AE_Tools.make_Clemens_decoder(latent_dim)

ms_Net = SDE_Tools.mu_sig_Net(M, latent_dim, n, act_ms_Net, SDE_Net_complexity, forceHigherOrder=forceHigherOrder)
reconstructor = SDE_Tools.make_Tensorwise_Reconstructor(latent_dim*pictureColors, latent_dim*pictureColors, n, T, frames-M+1, ms_Net, batch_size, applyBM=False)
#SDE_AE = SDE_Autoencoder(encoder, reconstructor, decoder)

rec_loss = AE_Tools.make_binary_crossentropy_rec_loss(M, frames)
ms_rec_loss = SDE_Tools.make_reconstruction_Loss(M, n, T, frames, batch_size, reconstructor, derivatives)
#p_loss = SDE_Tools.make_pointwise_Loss(T, frames, ms_Net, Model, getOwnInputs=True)
#cv_loss = SDE_Tools.make_covariance_Loss(T, frames, batch_size, ms_Net, Model, getOwnInputs=True)
#ss_loss = SDE_Tools.make_sigma_size_Loss(ms_Net, Model, getOwnInputs=True)
#ar_loss = SDE_Tools.make_anti_regularizer_Loss(ms_Net, Model)


#alpha = 0.5
#loss = lambda x_org,out1: 10*rec_loss(x_org, out1) + 1*ms_rec_loss(out3,out4) #+ alpha*p_loss(x_org) #+ 5*alpha*cv_loss(x_org) + alpha*ss_loss(x_org) #+ 2*ar_loss(x_org)
def CustomLoss(inputs, Z_enc_mean_List, Z_enc_log_var_List, Z_enc_List, Z_rec_List, X_rec_List):
    S = 10*rec_loss(inputs, X_rec_List)
    #S += 1*ms_rec_loss(Z_enc_List,None)
    S += 0.5*ms_rec_loss(Z_enc_List,Z_rec_List)
    return S



########################################################
# SDE_VAE definieren
'''
class SDE_Autoencoder(tf.keras.Model):
    def __init__(self, encoder, reconstructor, decoder):
        super(SDE_Autoencoder, self).__init__()
        self.encoder = encoder
        self.decoder = decoder
        self.reconstructor = reconstructor

    def fullcall(self, inputs):
        #print('SDE_AE input:',inputs.shape)
        frames_List = tf.split(inputs, len(inputs[0]), axis=1)
        Z_enc_List = []
        for x in frames_List:
            Z_enc = self.encoder(x[:,0,:,:,:])
            Z_enc_List.append(Z_enc)
            #print('Z_enc:',Z_enc.shape, inputs.shape)
        Z_enc_List = tf.stack(Z_enc_List,axis=1)
        #print('Z_enc_List:',Z_enc_List.shape)
        Z_0 = Z_enc_List[:,0,:]
        Z_rec_List = self.reconstructor(Z_0)
        #print('Z_rec_List:',Z_rec_List.shape)
        X_rec_List = []
        for i in range(frames-M+1):
            X_rec_List.append(self.decoder(Z_rec_List[:,i,:]))
        X_rec_List = tf.stack(X_rec_List, axis=1)
        #print('X_rec_List:',X_rec_List.shape)
        return [Z_enc_List, Z_rec_List, X_rec_List]

    def call(self, inputs):
        Z_enc_List, Z_rec_List, X_rec_List = self.fullcall(inputs)
        return X_rec_List
'''

class SDE_Variational_Autoencoder(tf.keras.Model):
    def __init__(self, encoder, reconstructor, decoder):
        super(SDE_Variational_Autoencoder, self).__init__()
        #self.outputs = ['a','b','c','d','e']
        self.encoder = encoder
        self.decoder = decoder
        self.reconstructor = reconstructor

    def fullcall(self, inputs):
        #inputs hat dim: batch_size x frames x pictureWidth x pictureHeight x pictureColors
        #print('fullcall on:',inputs, inputs.shape[1])
        frames_List = tf.split(inputs, inputs.shape[1], axis=1)
        Z_enc_List = []
        Z_enc_mean_List = []
        Z_enc_log_var_List = []
        for x in frames_List:
            #x hat dim: batch_size x 1 x pictureWidth x pictureHeight x pictureColors
            Z_enc_mean, Z_enc_log_var, Z_enc = self.encoder(x[:,0,:,:,:])
            Z_enc_List.append(tf.transpose([Z_enc], [1,0,2]))
            Z_enc_mean_List.append(tf.transpose([Z_enc_mean], [1,0,2]))
            Z_enc_log_var_List.append(tf.transpose([Z_enc_log_var], [1,0,2]))
        Z_enc_List = tf.keras.layers.Concatenate(axis=1)(Z_enc_List)
        #Z_enc_List hat dim: batch_size x frames x latent_dim
        Z_enc_mean_List = tf.keras.layers.Concatenate(axis=1)(Z_enc_mean_List)
        #Z_enc_mean_List hat dim: batch_size x frames x latent_dim
        Z_enc_log_var_List = tf.keras.layers.Concatenate(axis=1)(Z_enc_log_var_List)
        #Z_enc_log_var_List hat dim: batch_size x frames x latent_dim

        Z_derivatives = derivatives(Z_enc_List)
        #Z_derivatives hat dim: batch_size x frames-M+1 x M x latent_dim
        Z_derivatives_0 = Z_derivatives[:,0,:,:]
        #Z_derivatives_0 hat dim: batch_size x M x latent_dim

        #ACHTUNG: Mogeln
        #Z_rec_List = Z_enc_List[:,:frames-M+1,:]
        Z_rec_List = self.reconstructor(Z_derivatives_0)[:,:,0,:]
        #Z_rec_List = 0.5*self.reconstructor(Z_derivatives_0)[:,:,0,:] + 0.5*Z_enc_List[:,:frames-M+1,:]
        #Z_rec_List hat dim: batch_size x frames-M+1 x latent_dim

        X_rec_List = []
        for i in range(frames-M+1):
            X_rec = self.decoder(Z_rec_List[:,i,:])
            X_rec = tf.transpose([X_rec], [1,0,2,3,4])
            X_rec_List.append(X_rec)
        #X_rec_List = tf.stack(X_rec_List, axis=1, name = 'output_5')
        X_rec_List = tf.keras.layers.Concatenate(axis=1)(X_rec_List)
        #X_rec_List hat dim: batch_size x frames-M+1 x pictureWidth x pictureHeight x pictureColors
        return [Z_enc_mean_List, Z_enc_log_var_List, Z_enc_List, Z_rec_List, X_rec_List]

    def call(self, inputs):
        Z_enc_mean_List, Z_enc_log_var_List, Z_enc_List, Z_rec_List, X_rec_List = self.fullcall(inputs)

        return CustomLoss(inputs, Z_enc_mean_List, Z_enc_log_var_List, Z_enc_List, Z_rec_List, X_rec_List)



########################################################
# Model aufbauen und trainieren

#encoder = AE_Tools.LocalEncoder(latent_dim, M, pictureWidth, pictureHeight, pictureColors, act, complexity=CNN_complexity)
#decoder = AE_Tools.FramewiseDecoder(latent_dim, pictureWidth, pictureHeight, pictureColors, act, complexity=CNN_complexity)



SDE_VAE = SDE_Variational_Autoencoder(encoder, reconstructor, decoder)
Model = SDE_VAE
#inp hat dim: None x frames x pictureWidth x pictureHeight x pictureColors
#inp = tf.keras.layers.Input(shape=(frames, pictureWidth, pictureHeight, pictureColors))
#Model = tf.keras.Model(inputs=inp, outputs=[SDE_VAE.outp1(inp),SDE_VAE.outp2(inp),SDE_VAE.outp3(inp),SDE_VAE.outp4(inp),SDE_VAE(inp)])


Model.compile(optimizer='adam', loss= lambda x,arg:arg)#, metrics=[lambda x,out: 10*rec_loss(x,out)])
Model.fit(x_train, x_train, epochs=epochs, batch_size=batch_size, shuffle=False)




'''
AE = AE_Tools.SimpleAutoencoder(encoder, decoder)
AE.compile(optimizer='adam', loss=rec_loss)
AE.fit(x_train_longlist, x_train_longlist[:,:,:,1], epochs=epochs, batch_size=batch_size*(frames-M+1), shuffle=False)
'''
'''
#SDE-Datensatz erstellen
Z_org = Model.fullcall(x_train)[-3]
print('BBBBBBBBBB',Z_org.shape)
np.save('C:/Users/bende/Documents/Uni/SML-Projekt/SDE_Daten', Z_org)
'''

######################################

#x_test = data.create_dataset(dataset_size=100, frames=10, picture_size=28, object_number=3)
k = 0
x_test_org = x_train[0:batch_size]
print('x_test_org:',x_test_org.shape)
_,_,enc_lat,rec_lat,rec_imgs = Model.fullcall(x_train[0:batch_size,:,:,:,:])
print('rec_imgs:',rec_imgs.shape)
#enc_lat = list(map(lambda i: encoder(x_train[i,:,:,:,:])[-1], range(batch_size)))
#enc_lat = tf.stack(enc_lat, axis=0)

#Z_0 = derivatives(enc_lat)[:,0,:,:]
#rec_lat = reconstructor(Z_0)[:,:,0,:]

print('enc_lat:',enc_lat.shape)
#rec_lat = reconstructor(enc_lat[:,0,:])
print('rec_lat:',rec_lat.shape)

fig, axs = plt.subplots(9, 10)
for i in range(2):
    for j in range(10):
        axs[4*i, j].imshow(x_test_org[i,j,:,:,0], cmap='gray')
        axs[4*i+1, j].plot(np.linspace(1,latent_dim,latent_dim),enc_lat[i,j,:],'o')
        axs[4*i+2, j].plot(np.linspace(1,latent_dim,latent_dim),rec_lat[i,j,:],'o')
        axs[4*i+3, j].imshow(rec_imgs[i,j,:,:,0], cmap='gray')
for i in range(5):
    axs[8, 2*i].plot(np.linspace(1,frames,frames),enc_lat[i,:,:],'-')
    axs[8, 2*i+1].plot(np.linspace(1,frames-M+1,frames-M+1),rec_lat[i,:,:],'-')
plt.show()
