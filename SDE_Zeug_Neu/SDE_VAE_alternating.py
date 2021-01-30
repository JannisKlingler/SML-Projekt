import matplotlib.pyplot as plt
import numpy as np
import tensorflow as tf
import scipy as sp
from keras import backend as K

try:
    import SDE_Tools
    import AE_Tools
    import SDE_VAE_Tools
except:
    raise Exception('Could not load necessary Tools. Please execute file in its original location.')

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
latent_dim = 15  #5-30 sollte gut gehen
batch_size = 50  # eher klein halten, unter 100 falls möglich, 50 klappt gut
train_size = 60000
test_size = 10000 # wird hier noch nicht gebraucht
frames = 20  # Number of images in every datapoint. Choose accordingly to dataset size.
act_CNN = 'relu'  # Activation function 'tanh' is used in odenet.
act_ms_Net = 'tanh'
Time = 50  # number of seconds of the video
fps = Time/frames
n = 1
pictureWidth = 28
pictureHeight = 28
pictureColors = 1
M = 2 #für 2 ist es ein 2-SDE-VAE, M sollte nie größer als frames//2 sein
CNN_complexity = 20 #wird zur zeit garnicht verwenden
SDE_Net_complexity = 8*latent_dim # scheint mit 50 immer gut zu klappen
forceHigherOrder = False

VAE_epochs_starting = 5
SDE_epochs_starting = 20
expected_SDE_complexity = 20




data_path = 'C:/Users/bende/Documents/Uni/Datasets/'

# %%
########################################################
# Datensatz laden oder erstellen
try:
    #raise Exception('Ich will den Datensatz neu erstellen')
    x_train = np.load(data_path+'rotatingMNIST_train_{}frames.npy'.format(frames))
    x_test = np.load(data_path+'rotatingMNIST_test_{}frames.npy'.format(frames))
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
        np.save(data_path+'SDE_Zeug_Neu/rotatingMNIST_train', x_train)
        np.save(data_path+'SDE_Zeug_Neu/rotatingMNIST_test', x_test)
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

#encoder = AE_Tools.FramewiseEncoder(latent_dim, pictureWidth, pictureHeight, pictureColors, act_CNN, complexity=CNN_complexity, variational=True)
#decoder = AE_Tools.FramewiseDecoder(latent_dim, pictureWidth, pictureHeight, pictureColors, act_CNN, complexity=CNN_complexity)

encoder = AE_Tools.make_Clemens_encoder(latent_dim)
decoder = AE_Tools.make_Clemens_decoder(latent_dim)

ms_Net = SDE_Tools.mu_sig_Net(M, latent_dim, n, act_ms_Net, SDE_Net_complexity, forceHigherOrder=forceHigherOrder)
reconstructor = SDE_Tools.make_Tensorwise_Reconstructor(latent_dim*pictureColors, latent_dim*pictureColors, n, Time, frames-M+1, ms_Net, expected_SDE_complexity, applyBM=False)


rec_loss = AE_Tools.make_binary_crossentropy_rec_loss(frames)
ms_rec_loss = SDE_Tools.make_reconstruction_Loss(M, n, Time, frames, batch_size, reconstructor, derivatives)
p_loss = SDE_Tools.make_pointwise_Loss(M, latent_dim, Time, frames, ms_Net, expected_SDE_complexity)
cv_loss = SDE_Tools.make_covariance_Loss(latent_dim, Time, frames, batch_size, ms_Net, expected_SDE_complexity)
ss_loss = SDE_Tools.make_sigma_size_Loss(latent_dim, ms_Net)


def VAELoss(X_org, Z_enc_mean_List, Z_enc_log_var_List, Z_enc_List, Z_derivatives, Z_rec_List, X_rec_List):
    S = 20*rec_loss(X_org, X_rec_List)
    return S

def SDELoss(Z_derivatives, ms_rec):
    S = 0
    #S += alpha*1*ms_rec_loss(Z_enc_List,Z_rec_List)
    S += 10*p_loss(Z_derivatives,ms_rec)
    S += 1*cv_loss(Z_derivatives,ms_rec)
    #S += 1000*ss_loss(Z_derivatives,ms_rec) #mal ohne probieren
    return S
alpha = 0.5
def StartingLoss(X_org, Z_enc_mean_List, Z_enc_log_var_List, Z_enc_List, Z_derivatives, Z_rec_List, X_rec_List):
    S = 20*rec_loss(X_org, X_rec_List)
    S += 5*ms_rec_loss(Z_derivatives,Z_rec_List)
    #S += alpha*10*p_loss(Z_derivatives,None)
    #S += alpha*1*cv_loss(Z_derivatives,None)
    #S += beta*100*ss_loss(Z_derivatives,None)
    return S

beta = 0.2

def FullLoss(X_org, Z_enc_mean_List, Z_enc_log_var_List, Z_enc_List, Z_derivatives, Z_rec_List, X_rec_List):
    S = 20*rec_loss(X_org, X_rec_List)
    #S += alpha*1*ms_rec_loss(Z_enc_List,Z_rec_List)
    S += beta*10*p_loss(Z_derivatives,None)
    S += beta*1*cv_loss(Z_derivatives,None)
    #S += beta*1000*ss_loss(Z_derivatives,None)
    return S

########################################################
# SDE_VAE definieren
SDE_VAE = SDE_VAE_Tools.SDE_Variational_Autoencoder(M, 1, encoder, derivatives, reconstructor, decoder, StartingLoss)
#inp hat dim: None x frames x pictureWidth x pictureHeight x pictureColors

print('model defined')


########################################################
# Model ohne SDE-Rekonstruktion die latenten Darstellungen lernen lassen
print('initial training for encoder and decoder to learn a first latent representation')
SDE_VAE.reconstruct_smoothly = False
SDE_VAE.compile(optimizer='adam', loss= lambda x,arg:arg)
SDE_VAE.fit(x_train, x_train, epochs=VAE_epochs_starting, batch_size=batch_size, shuffle=False)
SDE_VAE.summary()


########################################################
# Die SDE-Rekonstruktion der latenten Darstellungen lernen lassen
# Dieses training ist merklich schneller auf der haupt-cpu ohne verwendung einer gpu
print('initial training to learn SDE governing latent representation')
with tf.device('/cpu:0'):
    ms_Net.compile(optimizer='adam', loss=SDELoss, metrics=[ss_loss, lambda x,m: ms_rec_loss(x,None)])
    _,_,_,z_train_derivatives,_,_ = SDE_VAE.fullcall(x_train)
    z_train_derivatives = tf.constant(z_train_derivatives)
    ms_Net.fit(z_train_derivatives, z_train_derivatives, epochs=SDE_epochs_starting, batch_size=batch_size, shuffle=False)
    ms_Net.summary()


########################################################
# En-&Decoder und SDE-Rekonstruktion zusammen trainieren
print('main training with SDEs and Decoders combined')
SDE_VAE.custom_loss = FullLoss
SDE_VAE.compile(optimizer='adam', loss= lambda x,arg:arg)
SDE_VAE.fit(x_train, x_train, epochs=epochs, batch_size=batch_size, shuffle=False)
SDE_VAE.summary()


########################################################
# Modell so einstellen, dass glatter reconstruiert wird.
SDE_VAE.reconstruct_smoothly = True


########################################################
# Ergebnisse speichern
_, _, Z_enc_List, _, Z_rec_List, X_rec_List = SDE_VAE.fullcall(x_test)

np.save(data_path+'Results_SDE_rotMNIST_Z_org_{}frames'.format(frames), x_test_path)
np.save(data_path+'Results_SDE_rotMNIST_X_org_{}frames'.format(frames), x_test)
np.save(data_path+'Results_SDE_rotMNIST_Z_enc_{}frames'.format(frames), Z_enc_List)
np.save(data_path+'Results_SDE_rotMNIST_Z_rec_{}frames'.format(frames), Z_rec_List)
np.save(data_path+'Results_SDE_rotMNIST_X_rec_{}frames'.format(frames), X_rec_List)


########################################################
# Ergebnisse darstellen

#x_test = data.create_dataset(dataset_size=100, frames=10, picture_size=28, object_number=3)
k = 0
x_test_org = x_train[2:batch_size]
print('x_test_org:',x_test_org.shape)
_,_,enc_lat,_,rec_lat,rec_imgs = SDE_VAE.fullcall(x_test_org)
print('rec_imgs:',rec_imgs.shape)
#enc_lat = list(map(lambda i: encoder(x_train[i,:,:,:,:])[-1], range(batch_size)))
#enc_lat = tf.stack(enc_lat, axis=0)

#Z_0 = derivatives(enc_lat)[:,0,:,:]
#rec_lat = reconstructor(Z_0)[:,:,0,:]

print('enc_lat:',enc_lat.shape)
#rec_lat = reconstructor(enc_lat[:,0,:])
print('rec_lat:',rec_lat.shape)

fig, axs = plt.subplots(9, 10)
for i in range(4):
    for j in range(10):
        axs[2*i, j].imshow(x_test_org[i,j,:,:,0], cmap='gray')
        axs[2*i+1, j].imshow(rec_imgs[i,j,:,:,0], cmap='gray')
for i in range(5):
    axs[8, 2*i].plot(np.linspace(1,frames,frames),enc_lat[i,:,:],'-')
    axs[8, 2*i+1].plot(np.linspace(1,frames-M+1,frames-M+1),rec_lat[i,:,:],'-')
plt.show()
