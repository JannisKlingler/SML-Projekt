#from trainSDE import mu_sig_Net
import time
import glob
import matplotlib.pyplot as plt
import numpy as np
import tensorflow as tf
import tensorflow_probability as tfp
from keras import backend as K
tfd = tfp.distributions
# import datasets as data


# Needed for gpu support on some machines
config = tf.compat.v1.ConfigProto(
    gpu_options=tf.compat.v1.GPUOptions(per_process_gpu_memory_fraction=0.8))
config.gpu_options.allow_growth = True
session = tf.compat.v1.Session(config=config)
tf.compat.v1.keras.backend.set_session(session)

########################################################
# %% hyperparameter
epochs = 4
latent_dim = 10  # Dimensionality for latent variables. 20-30 works fine.
batch_size = 100  # ≥100 as suggested by Kingma in Autoencoding Variational Bayes.
train_size = 60000  # Data points in train set. Choose accordingly to dataset size.
test_size = 10000  # Data points in test set. Choose accordingly to dataset size.
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


ode_integration = 'trivialsum'  # options: 'DormandPrince' , 'trivialsum'
dt = 0.1  # Step size for numerical integration. Increasing dt reduces training time
# but may impact predictive qualitiy. 'trivialsum'  uses dt = 1 and only reconstruction
# loss as training criteria. This is very fast and works surprisingly well.
data_path = 'C:/Users/Admin/Desktop/Python/Datasets/'
job = 'rotatingMNIST'  # Dataset for training. Options: 'rotatingMNIST' , 'bouncingBalls'

# %%
########################################################
# Datensatz laden oder erstellen
try:
    x_train = np.load(data_path + job + '_train.npy')
    x_test = np.load(data_path + job + '_test.npy')

except:
    print('Dataset is being generated. This may take a few minutes.')

    if job == 'rotatingMNIST':
        x_train, x_test, _ = data.create_dataset_rotatingMNIST(
            train_dataset_size=60000, test_dataset_size=10000, frames=10, variation=False)

    if job == 'bouncingBalls':
        x_train = data.create_dataset_bouncingBalls(dataset_size=60000, frames=10,
                                                    picture_size=28, object_number=3, variation=False, pictures=True)
        x_test = data.create_dataset_bouncingBalls(dataset_size=10000, frames=10,
                                                   picture_size=28, object_number=3, variation=False, pictures=True)

    np.save(data_path + job + '_train', x_train)
    np.save(data_path + job + '_test', x_test)
    print('Dataset generated')


# Dim: train_size x pictureWidth x pictureHeight*pictureColors x frames
x_train = x_train[0:train_size]
print('train-shape:', x_train.shape)


# Dim: test_size x frames x pictureWidth x pictureHeight x pictureColors
x_test = x_test[0:test_size]
x_test = np.transpose(np.array([x_test]), (1, 4, 2, 3, 0))

'''
########################################################
# Datensatz für Encoder erstellen

x_train = np.array(x_train)
train_size, frames, d = x_train.shape
for i in range(train_size):
    for j in range()


L = list(map(lambda i: np.concatenate(list(map(,))), range(train_size)))
L = list(map(lambda i: x_train[i,:,:,:,:], range(train_size)))

#Dim: train_size*(frames-2) x 3 x pictureWidth x pictureHeight x pictureColors
x_train_longlist = np.concatenate(L, axis=0)
'''

R = 20


class mu_sig_Net(tf.keras.Model):
    def __init__(self, d):
        self.inp = z = tf.keras.layers.Input(shape=(d))

        # Netzwerk um mu zu lernen
        mu = tf.keras.layers.Dense(R*d, activation=act)(z)
        mu = tf.keras.layers.Dense(R*d, activation=act)(mu)
        mu = tf.keras.layers.Dense(d, activation=act)(mu)
        mu = tf.keras.layers.Reshape((d, 1))(mu)

        # Netzwerk um sigma zu lernen
        sig = tf.keras.layers.Dense(R*d*n, activation=act)(z)
        sig = tf.keras.layers.Dense(R*d*n, activation=act)(sig)
        sig = tf.keras.layers.Dense(d*n, activation=act)(sig)
        sig = tf.keras.layers.Reshape((d, n))(sig)

        # Netzwerke zusammenfügen
        mu_sig = tf.keras.layers.Concatenate(axis=-1)([mu, sig])

        super(mu_sig_Net, self).__init__(
            self.inp, mu_sig, name="mu_sig_Net")


class PointwiseEncoder(tf.keras.Model):
    def __init__(self, latent_dim, nrFrames, pictureWidth, pictureHeight, pictureColors, act):
        self.inp = z = tf.keras.layers.Input(
            shape=(pictureWidth, pictureHeight*pictureColors, nrFrames))
        print('AA:', z)
        vel_enc = tf.keras.layers.Conv2D(32, (3, 3), padding="same",
                                         activation=act)(z)
        print('BB:', vel_enc)
        vel_enc = tf.keras.layers.MaxPooling2D((2, 2))(vel_enc)
        vel_enc = tf.keras.layers.Conv2D(64, (3, 3), activation=act)(vel_enc)
        vel_enc = tf.keras.layers.MaxPooling2D((2, 2))(vel_enc)
        vel_enc = tf.keras.layers.Conv2D(128, (3, 3), activation=act)(vel_enc)
        vel_enc = tf.keras.layers.Flatten()(vel_enc)

        v_μ = tf.keras.layers.Dense(latent_dim, activation=act)(vel_enc)
        v_log_σ = tf.keras.layers.Dense(latent_dim, activation=act)(vel_enc)

        v = tf.keras.layers.Lambda(lambda arg: arg[0] + K.exp(arg[1]) * K.random_normal(
            shape=(K.shape(arg[0])[0], latent_dim), mean=0.0, stddev=1.0))([v_μ, v_log_σ])

        super(PointwiseEncoder, self).__init__(
            self.inp, v, name="PointwiseEncoder")


P_enc = PointwiseEncoder(latent_dim, 3, pictureWidth, pictureHeight, pictureColors, act)
ms_Net = mu_sig_Net(latent_dim)

# Dim von x: train_size x pictureWidth x pictureHeight*pictureColors x frames


def encode_sequence(x):
    List = []
    for i in range(1, frames-1):
        encoded = P_enc(x[:, :, :, i-1:i+2])
        print('CCC:', encoded)
        List.append(encoded)
    List = tf.stack(List, axis=1)
    return List


x_train_enc = encode_sequence(x_train)

print('DDD:', np.array(x_train_enc).shape)


vae = tf.keras.Model(P_enc.inp, encode_seqence(P_enc.inp))


class ODE2_VAE_ConvTime_Encoder(tf.keras.Model):
    def __init__(self, frames, latent_dim, act):

        self.inp = x = tf.keras.layers.Input(shape=(28, 28, 10))

        first_frame = x[:, :, :, 0]
        first_frame = tf.keras.layers.Reshape((28, 28, 1))(first_frame)
        first_three_frames = x[:, :, :, 0:3]

        pos_enc = tf.keras.layers.Conv2D(16, (3, 3), padding="same", activation=act)(first_frame)
        pos_enc = tf.keras.layers.MaxPooling2D((2, 2))(pos_enc)
        pos_enc = tf.keras.layers.Conv2D(32, (3, 3), activation=act)(pos_enc)
        pos_enc = tf.keras.layers.MaxPooling2D((2, 2))(pos_enc)
        pos_enc = tf.keras.layers.Conv2D(64, (3, 3), activation=act)(pos_enc)
        pos_enc = tf.keras.layers.Flatten()(pos_enc)
        s_μ = tf.keras.layers.Dense(latent_dim, activation=act)(pos_enc)
        s_log_σ = tf.keras.layers.Dense(latent_dim, activation=act)(pos_enc)

        s = tf.keras.layers.Lambda(lambda arg: arg[0] + K.exp(arg[1]) * K.random_normal(
            shape=(K.shape(arg[0])[0], latent_dim), mean=0.0, stddev=1.0))([s_μ, s_log_σ])

        vel_enc = tf.keras.layers.Conv2D(32, (3, 3), padding="same",
                                         activation=act)(first_three_frames)
        vel_enc = tf.keras.layers.MaxPooling2D((2, 2))(vel_enc)
        vel_enc = tf.keras.layers.Conv2D(64, (3, 3), activation=act)(vel_enc)
        vel_enc = tf.keras.layers.MaxPooling2D((2, 2))(vel_enc)
        vel_enc = tf.keras.layers.Conv2D(128, (3, 3), activation=act)(vel_enc)
        vel_enc = tf.keras.layers.Flatten()(vel_enc)
        v_μ = tf.keras.layers.Dense(latent_dim, activation=act)(vel_enc)
        v_log_σ = tf.keras.layers.Dense(latent_dim, activation=act)(vel_enc)

        v = tf.keras.layers.Lambda(lambda arg: arg[0] + K.exp(arg[1]) * K.random_normal(
            shape=(K.shape(arg[0])[0], latent_dim), mean=0.0, stddev=1.0))([v_μ, v_log_σ])

        z = tf.stack([s, v], axis=1)

        sv_μ = tf.keras.layers.Concatenate(axis=1)([s_μ, v_μ])
        sv_σ = K.exp(tf.keras.layers.Concatenate(axis=1)([s_log_σ, v_log_σ]))
        super(ODE2_VAE_ConvTime_Encoder, self).__init__(
            self.inp, [sv_μ, sv_σ, z], name="Encoder")
