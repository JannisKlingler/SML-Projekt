import scipy as sp
from keras import backend as K
import matplotlib.pyplot as plt
from mpl_toolkits.axes_grid1 import ImageGrid
import numpy as np
import math as m
import tensorflow as tf
import tensorflow_probability as tfp
tfd = tfp.distributions

# %%

frames = 10


try:
    x_train = np.load('C:/Users/Admin/Desktop/Python/Datasets/rotatingMNIST_train.npy')
    x_test = np.load('C:/Users/Admin/Desktop/Python/Datasets/rotatingMNIST_test.npy')
except:
    print('Dataset is being generated. This may take a few minutes.')
    N = 100
    (x_train, y_train), (x_test, y_test) = tf.keras.datasets.mnist.load_data()
    x_test = x_test[0:N]
    x_train = x_train[0:N]
    x_train_rot = list(map(lambda b: list(map(lambda i: np.where(sp.ndimage.rotate(
        b, (i+1) * 360/frames, reshape=False) > 127.5, 1.0, 0.0).astype('float32'), range(frames))), x_train))
    x_test_rot = list(map(lambda b: list(map(lambda i: np.where(sp.ndimage.rotate(
        b, (i+1) * 360/frames, reshape=False) > 127.5, 1.0, 0.0).astype('float32'), range(frames))), x_test))
    for j in range(len(x_test_rot)):
        for i in np.random.choice(range(3, 10), 3, replace=False):
            x_test_rot[j][i] = np.zeros((28, 28))
    x_train = np.transpose(np.array(x_train_rot), [0, 2, 3, 1])
    x_test = np.transpose(np.array(x_test_rot), [0, 2, 3, 1])
    #np.save('C:/Users/Admin/Desktop/Python/Datasets/rotatingMNIST_train', x_train)
    #np.save('C:/Users/Admin/Desktop/Python/Datasets/rotatingMNIST_test', x_test)
    print('Dataset generated')


latent_dim = 25
batch_size = 100
epochs = 1
akt_fun = 'relu'

# %%


class ODE_VAE_ConvTime_Encoder(tf.keras.Model):
    def __init__(self, frames, latent_dim, act):

        self.inp = x = tf.keras.layers.Input(shape=(28, 28, 10))

        Ls_mu = []
        Ls_logsig = []
        Lv_mu = []
        Lv_logsig = []
        Ls = []
        Lv = []
        for i in range(frames):
            one_frame = x[:, :, :, i]
            one_frame = tf.keras.layers.Reshape((28, 28, 1))(one_frame)
            if (i == 0):
                three_frames = x[:, :, :, 0:3]
            elif (i == frames-1):
                three_frames = x[:, :, :, frames-3:frames]
            else:
                three_frames = x[:, :, :, i-1:i+1]

            # Position Encoder
            xs = tf.keras.layers.Conv2D(16, (3, 3), padding="same", activation=act)(one_frame)
            xs = tf.keras.layers.MaxPooling2D((2, 2))(xs)
            xs = tf.keras.layers.Conv2D(32, (3, 3), activation=act)(xs)
            xs = tf.keras.layers.MaxPooling2D((2, 2))(xs)
            xs = tf.keras.layers.Conv2D(64, (3, 3), activation=act)(xs)
            xs = tf.keras.layers.Flatten()(xs)
            s_mu = tf.keras.layers.Dense(latent_dim, activation=act)(xs)
            s_logsig = tf.keras.layers.Dense(latent_dim, activation=act)(xs)

            s = tf.keras.layers.Lambda(lambda arg: arg[0] + K.exp(arg[1]) * K.random_normal(
                shape=(K.shape(arg[0])[0], latent_dim), mean=0.0, stddev=1.0))([s_mu, s_logsig])

            # Momentum Encoder
            xv = tf.keras.layers.Conv2D(16, (3, 3), padding="same", activation=act)(three_frames)
            xv = tf.keras.layers.MaxPooling2D((2, 2))(xv)
            xv = tf.keras.layers.Conv2D(32, (3, 3), activation=act)(xv)
            xv = tf.keras.layers.MaxPooling2D((2, 2))(xv)
            xv = tf.keras.layers.Conv2D(64, (3, 3), activation=act)(xv)
            xv = tf.keras.layers.Flatten()(xv)
            v_mu = tf.keras.layers.Dense(latent_dim, activation=act)(xv)
            v_logsig = tf.keras.layers.Dense(latent_dim, activation=act)(xv)

            v = tf.keras.layers.Lambda(lambda arg: arg[0] + K.exp(arg[1]) * K.random_normal(
                shape=(K.shape(arg[0])[0], latent_dim), mean=0.0, stddev=1.0))([v_mu, v_logsig])

            Ls.append(s)
            Lv.append(v)
            Ls_mu.append(s_mu)
            Ls_logsig.append(s_logsig)
            Lv_mu.append(v_mu)
            Lv_logsig.append(v_logsig)

        sTensor = tf.stack(Ls)
        sTensor = tf.transpose(sTensor, perm=[1, 0, 2])
        vTensor = tf.stack(Lv)
        vTensor = tf.transpose(vTensor, perm=[1, 0, 2])

        z = tf.stack([sTensor, vTensor])
        z = tf.transpose(z, perm=[1, 0, 2, 3])

        super(ODE_VAE_ConvTime_Encoder, self).__init__(
            self.inp, [Ls_mu, Ls_logsig, Lv_mu, Lv_logsig, z], name="Encoder")


class Differential_Function(tf.keras.Model):
    def __init__(self, frames, latent_dim, act):

        self.inp = z = tf.keras.layers.Input(shape=(2, frames, latent_dim))
        f = tf.keras.layers.Flatten()(z)
        f = tf.keras.layers.Dense(100, activation='tanh')(f)
        f = tf.keras.layers.Dense(100, activation='tanh')(f)
        f = tf.keras.layers.Dense(frames*latent_dim, activation='tanh')(f)
        f = tf.keras.layers.Reshape((frames, latent_dim))(f)

        T = 1
        x_1 = tf.keras.layers.RepeatVector(frames)(z[:, 0, 0, :])
        a = K.cumsum(f, axis=1)
        a = tf.keras.layers.Subtract()([a, f])
        x_3 = K.cumsum(a, axis=1)
        x_3 = tf.keras.layers.Subtract()([x_3, a])
        x_3 = tf.keras.layers.Lambda(lambda arg: T*T*arg)(x_3)

        a = tf.keras.layers.RepeatVector(frames)(z[:, 1, 0, :])
        b = K.cumsum(a, axis=1)
        b = tf.keras.layers.Subtract()([b, a])
        x_2 = tf.keras.layers.Lambda(lambda arg: T*arg)(b)
        sLsg = tf.keras.layers.Add()([x_1, x_2, x_3])

        super(Differential_Function, self).__init__(self.inp, sLsg, name="DifferentialFunction")


class ODE_Bernoulli_ConvTime_Decoder(tf.keras.Model):
    def __init__(self, frames, latent_dim, act):
        self.inp = x = tf.keras.layers.Input(shape=(frames, latent_dim))
        L = []
        for i in range(frames):
            a = x[:, i, :]
            a = tf.keras.layers.Dense(64, activation=act)(a)
            a = tf.keras.layers.Dense(4 * 4 * 64, activation=act)(a)
            a = tf.keras.layers.Reshape((4, 4, 64))(a)
            a = tf.keras.layers.Conv2DTranspose(32, (3, 3), activation=act)(a)
            a = tf.keras.layers.UpSampling2D(size=(2, 2))(a)
            a = tf.keras.layers.Conv2DTranspose(16, (3, 3), activation=act)(a)
            a = tf.keras.layers.UpSampling2D(size=(2, 2))(a)
            a = tf.keras.layers.Conv2DTranspose(1, (3, 3), padding="same", activation='sigmoid')(a)
            L.append(a)
        outp = tf.keras.layers.Concatenate(axis=-1)(L)

        super(ODE_Bernoulli_ConvTime_Decoder, self).__init__(self.inp, outp, name="Decoder")

# %%


def Trivial_Loss(encoder, f, decoder, frames):
    x = encoder.inp
    x_rec = decoder(f(encoder(x)[-1]))
    return K.sum(frames * tf.keras.losses.binary_crossentropy(x, x_rec), axis=(1, 2))


#log_p_xi_zi = K.sum(-28 * tf.keras.losses.binary_crossentropy(x[:,:,:,i], x_rec[:,:,:,i]), axis=-1)
# p(z) wird bei ihm an der Stelle concat(Lv,Ls) ausgewertet statt bei concat(Lv_i,Ls_i)

T = 1
A = []
for i in range(latent_dim):
    a = np.zeros((frames, latent_dim)).astype('Float32')
    for j in range(frames):
        a[j, i] = T
    a = [np.zeros((frames, latent_dim)).astype('Float32'), a]
    A.append(a)
A = np.array(A)
# print('!!!!!!!!!!!!!!!!!!!')
# print(np.shape(A))


def Bernoulli_ODE_Loss(encoder, f, decoder, frames):
    x = encoder.inp
    Ls_mu, Ls_logsig, Lv_mu, Lv_logsig, z = encoder(x)
    sLsg = f(z)
    x_rec = decoder(sLsg)
    log_p_x_z = - K.sum(frames * tf.keras.losses.binary_crossentropy(x, x_rec), axis=(1, 2))
    log_pz = tfd.MultivariateNormalDiag(loc=tf.zeros(
        2*frames*latent_dim), scale_diag=tf.ones(2*frames*latent_dim)).log_prob(tf.keras.layers.Flatten()(z))
    #a = log_qz_0 = tfd.MultivariateNormalDiag(
    #    loc=tf.zeros(latent_dim), scale_diag=tf.ones(latent_dim)).log_prob(z[:, 0, 0, :] + z[:, 1, 0, :])
    print('CCCCCCCCCCCCCCCCCCCCCCCCCC')
    muLayer = tf.keras.layers.Concatenate(axis=1)([Ls_mu[0], Lv_mu[0]])
    logsigLayer = tf.keras.layers.Concatenate(axis=1)([Ls_logsig[0], Lv_logsig[0]])
    sigLayer = K.exp(logsigLayer)
    xLayer = tf.keras.layers.Concatenate(axis=1)([z[:, 0, 0, :], z[:, 1, 0, :]])
    print(muLayer)
    print(sigLayer)
    print(xLayer)

    #a = tfd.MultivariateNormalDiag(loc=(muLayer), scale_diag=(sigLayer)).log_prob(xLayer)
    #tfd.MultivariateNormalDiag(loc=(Lv_mu[0]), scale_diag=(Lv_logsig[0])).log_prob(z[:, 1, 0, :])




    log_qz_0 = tfd.MultivariateNormalDiag(loc=(muLayer), scale_diag=(sigLayer)).log_prob(xLayer)

    fn = f(z)

    Trace = 0
    for i in range(latent_dim):
        z_2 = z + tf.constant(A[i])
        Trace += f(z_2)[:, :, i]

    Int = K.cumsum(Trace, axis=1)
    Llogqz0 = []
    for k in range(frames):
        Llogqz0.append(log_qz_0)
    log_qz_0 = tf.stack(Llogqz0)
    log_qz_0 = tf.transpose(Llogqz0, perm=[1,0])
    #print('Fehler:')
    #print(log_qz_0)
    #print(Int)
    log_qz = log_qz_0 - Int
    #print(log_qz)
    #print(K.sum(log_qz, axis=1))
    ELBOneg = - log_p_x_z - log_pz + K.sum(log_qz, axis=1)
    #return - log_p_x_z - log_pz
    print('AAAAAAAAAAAAAAAAAAAAA')
    print(log_pz)
    print(log_pz[0])
    print(ELBOneg)
    print(ELBOneg[0])
    return ELBOneg


encoder = ODE_VAE_ConvTime_Encoder(frames, latent_dim, akt_fun)
f = Differential_Function(frames, latent_dim, akt_fun)
decoder = ODE_Bernoulli_ConvTime_Decoder(frames, latent_dim, akt_fun)
ode2vae = tf.keras.Model(encoder.inp, decoder(f(encoder(encoder.inp)[-1])))

#loss = Trivial_Loss(encoder, f, decoder, 10)
loss = Bernoulli_ODE_Loss(encoder, f, decoder, 10)
ode2vae.add_loss(loss)
ode2vae.compile(optimizer='adam')
# %%

x = x_train[0:4]
x_rec = x_train[1:5] / 2. + .1
z = [np.zeros(2*latent_dim*frames) for i in range(4)]
Ls_mu = Lv_mu = [np.zeros(latent_dim*frames) for i in range(4)]
Ls_logsig = Lv_logsig = [np.ones(latent_dim*frames) for i in range(4)]


log_p_x_z = - K.sum(frames * tf.keras.losses.binary_crossentropy(x, x_rec), axis=(1, 2))
log_pz = tfd.MultivariateNormalDiag(loc=tf.zeros(
    2*frames*latent_dim), scale_diag=tf.ones(2*frames*latent_dim)).log_prob(z)

print('BBBBBBBBBBBBBBBB')
print(log_p_x_z)
print(log_p_x_z.numpy())
print(log_pz.numpy())
# %%
ode2vae.fit(x_train, epochs=epochs, batch_size=batch_size)

# %%
k = 0
rec_imgs = ode2vae.predict(x_test)
fig, index = plt.figure(figsize=(10, 10)), np.random.randint(len(x_test), size=5)
grid = ImageGrid(fig, 111,  nrows_ncols=(10, 10), axes_pad=0.1,)
plot = [x_test[index[0]][:, :, j] for j in range(10)]
for i in index:
    if k != 0:
        original = [x_test[i][:, :, j] for j in range(10)]
        plot = np.vstack((plot, original))
    reconst = [rec_imgs[i][:, :, j] for j in range(10)]
    plot = np.vstack((plot, reconst))
    k += 1
for ax, im in zip(grid, plot):
    plt.gray()
    ax.imshow(im)
plt.show()
