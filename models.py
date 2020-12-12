import tensorflow as tf
from keras import backend as K


# class Id_Middle(tf.keras.Model):
#    def __init__(self):
#        super(Id_Middle, self).__init__(self.inp, self.inp, name="Middle")
#        self.summary()


class ODE_VAE_ConvTime_Encoder(tf.keras.Model):
    def __init__(self, frames, latent_dim, act):
        self.inp = x = tf.keras.Input(shape=(28, 28, 10))

        # s_0, v_0 finden
        first_frame = x[:, :, :, 0]
        first_frame = tf.keras.layers.Reshape((28, 28, 1))(first_frame)
        first_3frames = x[:, :, :, 0:3]

        # Position Encoder
        xs = tf.keras.layers.Conv2D(16, (3, 3), padding="same", activation=act)(first_frame)
        xs = tf.keras.layers.Flatten()(xs)
        s_0_mu = tf.keras.layers.Dense(latent_dim, activation=act)(xs)
        s_0_logsig = tf.keras.layers.Dense(latent_dim, activation=act)(xs)

        s_0 = tf.keras.layers.Lambda(lambda arg: arg[0] + K.exp(arg[1]) * K.random_normal(
            shape=(K.shape(arg[0])[0], latent_dim), mean=0.0, stddev=1.0))([s_0_mu, s_0_logsig])

        # Momentum Encoder
        xv = tf.keras.layers.Conv2D(16, (3, 3), padding="same", activation=act)(first_3frames)
        xv = tf.keras.layers.Flatten()(xv)
        v_0_mu = tf.keras.layers.Dense(latent_dim, activation=act)(xv)
        v_0_logsig = tf.keras.layers.Dense(latent_dim, activation=act)(xv)

        v_0 = tf.keras.layers.Lambda(lambda arg: arg[0] + K.exp(arg[1]) * K.random_normal(
            shape=(K.shape(arg[0])[0], latent_dim), mean=0.0, stddev=1.0))([v_0_mu, v_0_logsig])

        '''
        # z_1,...,z_N finden
        x = tf.keras.layers.Conv2D(16, (3, 3), padding="same", activation=act)(x)
        x = tf.keras.layers.MaxPooling2D((2, 2))(x)
        x = tf.keras.layers.Conv2D(32, (3, 3), activation=act)(x)
        x = tf.keras.layers.MaxPooling2D((2, 2))(x)
        x = tf.keras.layers.Conv2D(64, (3, 3), activation=act)(x)
        x = tf.keras.layers.Flatten()(x)
        x = tf.keras.layers.Dense(64, activation=act)(x)

        s_mu = tf.keras.layers.Dense(frames, name="mu")(x)
        s_log_sig = tf.keras.layers.Dense(frames, name="log_sig")(x)

        v_mu = tf.keras.layers.Dense(frames, name="mu")(x)
        v_log_sig = tf.keras.layers.Dense(frames, name="log_sig")(x)

        s = tf.keras.layers.Lambda(lambda arg: arg[0] + K.exp(arg[1]) * K.random_normal(
            shape=(K.shape(arg[0])[0], frames), mean=0.0, stddev=1.0))([s_mu, s_log_sig])
        v = tf.keras.layers.Lambda(lambda arg: arg[0] + K.exp(arg[1]) * K.random_normal(
            shape=(K.shape(arg[0])[0], frames), mean=0.0, stddev=1.0))([v_mu, v_log_sig])
        '''

        # s_0 = 1 x latent_dim
        # x_1 = frames x latent_dim

        # Differential Function
        T = 1  # Zeit zwischen Frames
        #f = tf.keras.layers.Concatenate()([s, v])
        f = tf.keras.layers.Flatten()(self.inp)
        f = tf.keras.layers.Dense(64, activation='tanh')(f)
        f = tf.keras.layers.Dense(frames*latent_dim, activation='tanh')(f)
        f = tf.keras.layers.Reshape((frames, latent_dim))(f)
        x_1 = tf.keras.layers.RepeatVector(frames)(s_0)

        a = K.cumsum(f, axis=1)
        a = tf.keras.layers.Subtract()([a, f])
        x_3 = K.cumsum(a, axis=1)
        x_3 = tf.keras.layers.Subtract()([x_3, a])
        x_3 = tf.keras.layers.Lambda(lambda arg: T*T*arg)(x_3)

        a = tf.keras.layers.RepeatVector(frames)(v_0)
        b = K.cumsum(a, axis=1)
        b = tf.keras.layers.Subtract()([b, a])
        x_2 = tf.keras.layers.Lambda(lambda arg: T*arg)(b)

        sLsg = tf.keras.layers.Add()([x_1, x_2, x_3])
        # sLsg ist die lösung der DifferentialGl.

        super(ODE_VAE_ConvTime_Encoder, self).__init__(
            self.inp, [s_0_mu, s_0_logsig, v_0_mu, v_0_logsig, sLsg], name="Encoder")
        self.summary()


class ODE_Bernoulli_ConvTime_Decoder(tf.keras.Model):
    def __init__(self, frames, latent_dim, act):
        self.inp = x = tf.keras.Input(shape=(frames, latent_dim,))
        l = []
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
            l.append(a)
        outp = tf.keras.layers.Concatenate(axis=-1)(l)

        super(ODE_Bernoulli_ConvTime_Decoder, self).__init__(self.inp, [outp, outp], name="Decoder")
        self.summary()


class VAE_Dense_Encoder(tf.keras.Model):
    def __init__(self, latent_dim, act):
        dense_struc = [784, 500, 500, latent_dim]
        dropout = 0.1
        l = len(dense_struc) - 1
        self.inp = x = tf.keras.Input(shape=(28, 28, 1))
        x = tf.keras.layers.Reshape((dense_struc[0],))(x)

        for i in range(1, l):
            x = tf.keras.layers.Dense(dense_struc[i], activation=act)(x)
            if dropout != 0:
                x = tf.keras.layers.Dropout(dropout)(x)

        μ = tf.keras.layers.Dense(dense_struc[l], name="mu")(x)
        log_σ = tf.keras.layers.Dense(dense_struc[l], name="log_sig")(x)

        z = tf.keras.layers.Lambda(lambda arg: arg[0] + K.exp(arg[1]) * K.random_normal(
            shape=(K.shape(arg[0])[0], dense_struc[l]), mean=0.0, stddev=1.0))([μ, log_σ])

        super(VAE_Dense_Encoder, self).__init__(self.inp, [μ, log_σ, z], name="Encoder")
        self.summary()


class VAE_Conv_Encoder(tf.keras.Model):
    def __init__(self, latent_dim, act):
        self.inp = x = tf.keras.Input(shape=(28, 28, 1))
        x = tf.keras.layers.Conv2D(32, (3, 3), padding="same", activation=act)(x)
        x = tf.keras.layers.MaxPooling2D((2, 2))(x)
        x = tf.keras.layers.Conv2D(64, (3, 3), activation=act)(x)
        x = tf.keras.layers.MaxPooling2D((2, 2))(x)
        x = tf.keras.layers.Conv2D(128, (3, 3), activation=act)(x)
        x = tf.keras.layers.Flatten()(x)
        x = tf.keras.layers.Dense(128, activation='relu')(x)

        μ = tf.keras.layers.Dense(latent_dim, name="mu")(x)
        log_σ = tf.keras.layers.Dense(latent_dim, name="log_sig")(x)

        z = tf.keras.layers.Lambda(lambda arg: arg[0] + K.exp(arg[1]) * K.random_normal(
            shape=(K.shape(arg[0])[0], latent_dim), mean=0.0, stddev=1.0))([μ, log_σ])

        super(VAE_Conv_Encoder, self).__init__(self.inp, [μ, log_σ, z], name="Encoder")
        self.summary()


class VAE_ConvTime_Encoder(tf.keras.Model):
    def __init__(self, frames, latent_dim, act):
        self.inp = x = tf.keras.Input(shape=(28, 28, frames))
        x = tf.keras.layers.Conv2D(64, (3, 3), padding="same", activation=act)(x)
        x = tf.keras.layers.MaxPooling2D((2, 2))(x)
        x = tf.keras.layers.Conv2D(128, (3, 3), activation=act)(x)
        x = tf.keras.layers.MaxPooling2D((2, 2))(x)
        x = tf.keras.layers.Conv2D(256, (3, 3), activation=act)(x)
        x = tf.keras.layers.Flatten()(x)
        x = tf.keras.layers.Dense(256, activation=act)(x)

        μ = tf.keras.layers.Dense(latent_dim, name="mu")(x)
        log_σ = tf.keras.layers.Dense(latent_dim, name="log_sig")(x)

        z = tf.keras.layers.Lambda(lambda arg: arg[0] + K.exp(arg[1]) * K.random_normal(
            shape=(K.shape(arg[0])[0], latent_dim), mean=0.0, stddev=1.0))([μ, log_σ])

        super(VAE_ConvTime_Encoder, self).__init__(self.inp, [μ, log_σ, z], name="Encoder")
        self.summary()


class Bernoulli_Dense_Decoder(tf.keras.Model):
    def __init__(self, latent_dim, act):
        decoder_struc = [latent_dim, 500, 500, 784]
        dropout = 0.1
        l = len(decoder_struc) - 1
        self.inp = x = tf.keras.Input(shape=(decoder_struc[0],))

        for i in range(1, l):
            x = tf.keras.layers.Dense(decoder_struc[i], activation=act)(x)
            if dropout != 0:
                x = tf.keras.layers.Dropout(dropout)(x)

        x = tf.keras.layers.Dense(decoder_struc[l], activation='sigmoid')(x)
        outp = tf.keras.layers.Reshape((28, 28, 1))(x)
        super(Bernoulli_Dense_Decoder, self).__init__(self.inp, [outp, outp], name="Decoder")
        self.summary()


class Gauss_Conv_Decoder(tf.keras.Model):
    def __init__(self, latent_dim, act):
        self.inp = x = tf.keras.Input(shape=(latent_dim,))
        x = tf.keras.layers.Dense(128, activation=act)(x)
        x = tf.keras.layers.Dense(4 * 4 * 128, activation=act)(x)
        x = tf.keras.layers.Reshape((4, 4, 128))(x)
        x = tf.keras.layers.Conv2DTranspose(64, (3, 3), activation=act)(x)
        x = tf.keras.layers.UpSampling2D(size=(2, 2))(x)
        x = tf.keras.layers.Conv2DTranspose(32, (3, 3), activation=act)(x)
        x = tf.keras.layers.UpSampling2D(size=(2, 2))(x)

        μ = tf.keras.layers.Conv2DTranspose(1, (3, 3), padding="same")(x)
        log_σ = tf.keras.layers.Conv2DTranspose(1, (3, 3), padding="same", activation='sigmoid')(x)

        μ_reshape = tf.keras.layers.Reshape((784,))(μ)
        log_σ_reshape = tf.keras.layers.Reshape((784,))(log_σ)

        super(Gauss_Conv_Decoder, self).__init__(
            self.inp, [μ, μ_reshape, log_σ_reshape], name="Decoder")
        self.summary()


class Bernoulli_Conv_Decoder(tf.keras.Model):
    def __init__(self, latent_dim, act):
        self.inp = x = tf.keras.Input(shape=(latent_dim,))
        x = tf.keras.layers.Dense(128, activation=act)(x)
        x = tf.keras.layers.Dense(4 * 4 * 128, activation=act)(x)
        x = tf.keras.layers.Reshape((4, 4, 128))(x)
        x = tf.keras.layers.Conv2DTranspose(64, (3, 3), activation=act)(x)
        x = tf.keras.layers.UpSampling2D(size=(2, 2))(x)
        x = tf.keras.layers.Conv2DTranspose(32, (3, 3), activation=act)(x)
        x = tf.keras.layers.UpSampling2D(size=(2, 2))(x)
        outp = tf.keras.layers.Conv2DTranspose(1, (3, 3), padding="same", activation='sigmoid')(x)
        super(Bernoulli_Conv_Decoder, self).__init__(self.inp, [outp, outp], name="Decoder")
        self.summary()


class Bernoulli_ConvTime_Decoder(tf.keras.Model):
    def __init__(self, frames, latent_dim, act):
        self.inp = x = tf.keras.Input(shape=(latent_dim,))
        x = tf.keras.layers.Dense(256, activation=act)(x)
        x = tf.keras.layers.Dense(4 * 4 * 256, activation=act)(x)
        x = tf.keras.layers.Reshape((4, 4, 256))(x)
        x = tf.keras.layers.Conv2DTranspose(128, (3, 3), activation=act)(x)
        x = tf.keras.layers.UpSampling2D(size=(2, 2))(x)
        x = tf.keras.layers.Conv2DTranspose(64, (3, 3), activation=act)(x)
        x = tf.keras.layers.UpSampling2D(size=(2, 2))(x)
        outp = tf.keras.layers.Conv2DTranspose(
            frames, (3, 3), padding="same", activation='sigmoid')(x)
        super(Bernoulli_ConvTime_Decoder, self).__init__(self.inp, [outp, outp], name="Decoder")
        self.summary()


class Gauss_ConvTime_Decoder(tf.keras.Model):
    def __init__(self, frames, latent_dim, act):
        self.inp = x = tf.keras.Input(shape=(latent_dim,))
        x = tf.keras.layers.Dense(256, activation=act)(x)
        x = tf.keras.layers.Dense(4 * 4 * 256, activation=act)(x)
        x = tf.keras.layers.Reshape((4, 4, 256))(x)
        x = tf.keras.layers.Conv2DTranspose(128, (3, 3), activation=act)(x)
        x = tf.keras.layers.UpSampling2D(size=(2, 2))(x)
        x = tf.keras.layers.Conv2DTranspose(64, (3, 3), activation=act)(x)
        x = tf.keras.layers.UpSampling2D(size=(2, 2))(x)

        μ = tf.keras.layers.Conv2DTranspose(frames, (3, 3), padding="same")(x)
        log_σ = tf.keras.layers.Conv2DTranspose(
            frames, (3, 3), padding="same", activation='sigmoid')(x)

        μ_reshape = tf.keras.layers.Reshape((784*frames,))(μ)
        log_σ_reshape = tf.keras.layers.Reshape((784*frames,))(log_σ)

        super(Gauss_ConvTime_Decoder, self).__init__(
            self.inp, [μ, μ_reshape, log_σ_reshape], name="Decoder")
        self.summary()
