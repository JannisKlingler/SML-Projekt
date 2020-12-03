import tensorflow as tf
from keras import backend as K


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
    def __init__(self, latent_dim, act):
        self.inp = x = tf.keras.Input(shape=(28, 28, 10))
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

        μ = tf.keras.layers.Conv2DTranspose(1, (4, 4), padding="same")(x)
        log_σ = tf.keras.layers.Conv2DTranspose(1, (4, 4), padding="same", activation='relu')(x)

        μ_reshape = tf.keras.layers.Reshape((784,))(μ)
        log_σ_reshape = tf.keras.layers.Reshape((784,))(log_σ)

#        rec = tf.keras.layers.Lambda(lambda arg: arg[0] + K.exp(arg[1]) * K.random_normal(
#            shape=(K.shape(arg[0])[0], 784), mean=0.0, stddev=1.0))([μ_reshape, log_σ_reshape])

#        outp = tf.keras.layers.Reshape((28, 28, 1))(rec)

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
        outp = tf.keras.layers.Conv2DTranspose(1, (4, 4), padding="same", activation='sigmoid')(x)
        super(Bernoulli_Conv_Decoder, self).__init__(self.inp, [outp, outp], name="Decoder")
        self.summary()


class Bernoulli_ConvTime_Decoder(tf.keras.Model):
    def __init__(self, latent_dim, act):
        self.inp = x = tf.keras.Input(shape=(latent_dim,))
        x = tf.keras.layers.Dense(256, activation=act)(x)
        x = tf.keras.layers.Dense(4 * 4 * 256, activation=act)(x)
        x = tf.keras.layers.Reshape((4, 4, 256))(x)
        x = tf.keras.layers.Conv2DTranspose(128, (3, 3), activation=act)(x)
        x = tf.keras.layers.UpSampling2D(size=(2, 2))(x)
        x = tf.keras.layers.Conv2DTranspose(64, (3, 3), activation=act)(x)
        x = tf.keras.layers.UpSampling2D(size=(2, 2))(x)
        outp = tf.keras.layers.Conv2DTranspose(10, (4, 4), padding="same", activation='sigmoid')(x)
        super(Bernoulli_ConvTime_Decoder, self).__init__(self.inp, [outp, outp], name="Decoder")
        self.summary()


class Gauss_ConvTime_Decoder(tf.keras.Model):
    def __init__(self, latent_dim, act):
        self.inp = x = tf.keras.Input(shape=(latent_dim,))
        x = tf.keras.layers.Dense(256, activation=act)(x)
        x = tf.keras.layers.Dense(4 * 4 * 256, activation=act)(x)
        x = tf.keras.layers.Reshape((4, 4, 256))(x)
        x = tf.keras.layers.Conv2DTranspose(128, (3, 3), activation=act)(x)
        x = tf.keras.layers.UpSampling2D(size=(2, 2))(x)
        x = tf.keras.layers.Conv2DTranspose(64, (3, 3), activation=act)(x)
        x = tf.keras.layers.UpSampling2D(size=(2, 2))(x)

        μ = tf.keras.layers.Conv2DTranspose(10, (4, 4), padding="same")(x)
        log_σ = tf.keras.layers.Conv2DTranspose(10, (4, 4), padding="same", activation='relu')(x)

        μ_reshape = tf.keras.layers.Reshape((7840,))(μ)
        log_σ_reshape = tf.keras.layers.Reshape((7840,))(log_σ)

#        rec = tf.keras.layers.Lambda(lambda arg: arg[0] + K.exp(arg[1]) * K.random_normal(
#            shape=(K.shape(arg[0])[0], 7840), mean=0.0, stddev=1.0))([μ_reshape, log_σ_reshape])

        super(Gauss_ConvTime_Decoder, self).__init__(
            self.inp, [μ, μ_reshape, log_σ_reshape], name="Decoder")
        self.summary()
