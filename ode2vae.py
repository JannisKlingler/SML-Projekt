import bouncing_balls_clemens as data  # Für bouncing balls
import scipy as sp
from keras import backend as K
import matplotlib.pyplot as plt
from mpl_toolkits.axes_grid1 import ImageGrid
import numpy as np
import math as m
import tensorflow as tf
import tensorflow_probability as tfp
tfd = tfp.distributions


frames = 10
epochs = 2
latent_dim = 25
akt_fun = 'relu'

# Für rotating MNIST:
# try:
x_train = np.load('C:/Users/Admin/Desktop/Python/Datasets/rotatingMNIST_train.npy')
x_test = np.load('C:/Users/Admin/Desktop/Python/Datasets/rotatingMNIST_test.npy')

# except:
#    print('Dataset is being generated. This may take a few minutes.')
#    (x_train, y_train), (x_test, y_test) = tf.keras.datasets.mnist.load_data()
#    N = 1000
#    x_train = x_train[0:N]
#    x_test = x_test[0:N]
#    x_train_rot = list(map(lambda b: list(map(lambda i: np.where(sp.ndimage.rotate(
#        b, (i+1) * 360/frames, reshape=False) > 127.5, 1.0, 0.0).astype('float32'), range(frames))), x_train))
#    x_test_rot = list(map(lambda b: list(map(lambda i: np.where(sp.ndimage.rotate(
#        b, (i+1) * 360/frames, reshape=False) > 127.5, 1.0, 0.0).astype('float32'), range(frames))), x_test))
#    for j in range(len(x_test_rot)):
#        for i in np.random.choice(range(3, 10), 3, replace=False):
#            x_test_rot[j][i] = np.zeros((28, 28))
#    x_train = np.transpose(np.array(x_train_rot), [0, 2, 3, 1])
#    x_test = np.transpose(np.array(x_test_rot), [0, 2, 3, 1])
# np.save('C:/Users/Admin/Desktop/Python/Datasets/rotatingMNIST_train', x_train)
# np.save('C:/Users/Admin/Desktop/Python/Datasets/rotatingMNIST_test', x_test)
#    print('Dataset generated')


class ODE_VAE_ConvTime_Encoder(tf.keras.Model):
    def __init__(self, frames, latent_dim, act):

        self.inp = x = tf.keras.layers.Input(shape=(28, 28, 10))

        Ls_μ = []
        Ls_log_σ = []
        Lv_μ = []
        Lv_log_σ = []
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
            s_μ = tf.keras.layers.Dense(latent_dim, activation=act)(xs)
            s_log_σ = tf.keras.layers.Dense(latent_dim, activation=act)(xs)

            s = tf.keras.layers.Lambda(lambda arg: arg[0] + K.exp(arg[1]) * K.random_normal(
                shape=(K.shape(arg[0])[0], latent_dim), mean=0.0, stddev=1.0))([s_μ, s_log_σ])

            # Momentum Encoder
            xv = tf.keras.layers.Conv2D(16, (3, 3), padding="same", activation=act)(three_frames)
            xv = tf.keras.layers.MaxPooling2D((2, 2))(xv)
            xv = tf.keras.layers.Conv2D(32, (3, 3), activation=act)(xv)
            xv = tf.keras.layers.MaxPooling2D((2, 2))(xv)
            xv = tf.keras.layers.Conv2D(64, (3, 3), activation=act)(xv)
            xv = tf.keras.layers.Flatten()(xv)
            v_μ = tf.keras.layers.Dense(latent_dim, activation=act)(xv)
            v_log_σ = tf.keras.layers.Dense(latent_dim, activation=act)(xv)

            v = tf.keras.layers.Lambda(lambda arg: arg[0] + K.exp(arg[1]) * K.random_normal(
                shape=(K.shape(arg[0])[0], latent_dim), mean=0.0, stddev=1.0))([v_μ, v_log_σ])

            Ls.append(s)
            Lv.append(v)
            Ls_μ.append(s_μ)
            Ls_log_σ.append(s_log_σ)
            Lv_μ.append(v_μ)
            Lv_log_σ.append(v_log_σ)

        sTensor = tf.stack(Ls)
        sTensor = tf.transpose(sTensor, perm=[1, 0, 2])
        vTensor = tf.stack(Lv)
        vTensor = tf.transpose(vTensor, perm=[1, 0, 2])

        z = tf.stack([sTensor, vTensor])
        z = tf.transpose(z, perm=[1, 0, 2, 3])

        s_v_μ_0 = tf.keras.layers.Concatenate(axis=1)([Ls_μ[0], Lv_μ[0]])
        s_v_σ_0 = K.exp(tf.keras.layers.Concatenate(axis=1)([Ls_log_σ[0], Lv_log_σ[0]]))

        super(ODE_VAE_ConvTime_Encoder, self).__init__(
            self.inp, [s_v_μ_0, s_v_σ_0, z], name="Encoder")


class Differential_Function(tf.keras.Model):
    def __init__(self, frames, latent_dim, act):

        self.inp = z = tf.keras.layers.Input(shape=(2, frames, latent_dim))
        f = tf.keras.layers.Flatten()(z)
        f = tf.keras.layers.Dense(100, activation='tanh')(f)
        f = tf.keras.layers.Dense(100, activation='tanh')(f)
        f = tf.keras.layers.Dense(frames*latent_dim, activation='tanh')(f)
        f = tf.keras.layers.Reshape((frames, latent_dim))(f)

        𝞃 = 1 / frames
        x_1 = tf.keras.layers.RepeatVector(frames)(z[:, 0, 0, :])
        a = K.cumsum(f, axis=1)
        a = tf.keras.layers.Subtract()([a, f])
        x_3 = K.cumsum(a, axis=1)
        x_3 = tf.keras.layers.Subtract()([x_3, a])
        x_3 = tf.keras.layers.Lambda(lambda arg: 𝞃**2*arg)(x_3)

        a = tf.keras.layers.RepeatVector(frames)(z[:, 1, 0, :])
        b = K.cumsum(a, axis=1)
        b = tf.keras.layers.Subtract()([b, a])
        x_2 = tf.keras.layers.Lambda(lambda arg: 𝞃*arg)(b)
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


def Trivial_Loss(encoder, f, decoder, frames):
    x = encoder.inp
    x_rec = decoder(f(encoder(x)[-1]))
    return K.sum(frames * tf.keras.losses.binary_crossentropy(x, x_rec), axis=(1, 2))


𝞃 = 1 / frames
A = []
for i in range(latent_dim):
    a = np.zeros((frames, latent_dim))
    for j in range(frames):
        a[j, i] = 𝞃
    a = [np.zeros((frames, latent_dim)), a]
    A.append(a)
A = np.array(A).astype('Float32')


def Bernoulli_ODE_Loss(encoder, f, decoder, frames):
    x = encoder.inp
    s_v_μ_0, s_v_σ_0, z = encoder(x)
    sLsg = f(z)
    x_rec = decoder(sLsg)
    a = frames * tf.keras.losses.binary_crossentropy(x, x_rec)
    log_p_x_z = - K.sum(a, axis=(1, 2))
    log_pz = tfd.MultivariateNormalDiag(loc=tf.zeros(
        2*frames*latent_dim), scale_diag=tf.ones(2*frames*latent_dim)).log_prob(tf.keras.layers.Flatten()(z))

    diag_eval = tf.keras.layers.Concatenate(axis=1)([z[:, 0, 0, :], z[:, 1, 0, :]])
    log_qz_0 = tfd.MultivariateNormalDiag(loc=(s_v_μ_0), scale_diag=(s_v_σ_0)).log_prob(diag_eval)

    Trace = 0
    for i in range(latent_dim):
        z_2 = z + tf.constant(A[i])
        f_2 = f(z_2)
        Trace += f_2[:, :, i] - sLsg[:, :, i]
    Int = K.cumsum(Trace, axis=1)

    log_qz = frames * log_qz_0 - K.sum(Int, axis=1)

    ode_regul = 0.01 * np.sum([np.sum(f.get_weights()[i] ** 2)
                               for i in range(len(f.get_weights()))])

    ELBO = - ode_regul + log_p_x_z + log_pz - log_qz_0 - log_qz
    return - ELBO


encoder = ODE_VAE_ConvTime_Encoder(frames, latent_dim, akt_fun)
f = Differential_Function(frames, latent_dim, akt_fun)
decoder = ODE_Bernoulli_ConvTime_Decoder(frames, latent_dim, akt_fun)
ode2vae = tf.keras.Model(encoder.inp, decoder(f(encoder(encoder.inp)[-1])))

loss = Bernoulli_ODE_Loss(encoder, f, decoder, frames=frames)
ode2vae.add_loss(loss)
ode2vae.compile(optimizer='adam')

# Für bouncing balls
train_generator = data.DataGenerator(object_number=3, picture_size=28,
                                     frames=frames, dataset_size=1000, batch_size=100)

ode2vae.fit(train_generator, epochs=epochs, callbacks=[data.rec_loss_BouncingBalls()])

# Für rotating MNIST:
#ode2vae.fit(x_train, epochs=epochs, batch_size=100)


x_test = data.create_dataset(dataset_size=200, frames=frames)
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
