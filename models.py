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
        x = tf.keras.layers.Conv2D(32, (3, 3),
                                   strides=(2, 2), padding="same", activation=act)(x)
        x = tf.keras.layers.Conv2D(64, (3, 3), strides=(2, 2), padding="same", activation=act)(x)
        x = tf.keras.layers.Conv2D(128, (3, 3), strides=(2, 2), activation=act)(x)
        x = tf.keras.layers.Flatten()(x)
        x = tf.keras.layers.Dense(128, activation=act)(x)

        μ = tf.keras.layers.Dense(latent_dim, name="mu")(x)
        log_σ = tf.keras.layers.Dense(latent_dim, name="log_sig")(x)

        z = tf.keras.layers.Lambda(lambda arg: arg[0] + K.exp(arg[1]) * K.random_normal(
            shape=(K.shape(arg[0])[0], latent_dim), mean=0.0, stddev=1.0))([μ, log_σ])

        super(VAE_Conv_Encoder, self).__init__(self.inp, [μ, log_σ, z], name="Encoder")
        self.summary()


class VAE_ConvTime_Encoder(tf.keras.Model):
    def __init__(self, latent_dim, act):
        self.inp = x = tf.keras.Input(shape=(28, 28, 10))
        x = tf.keras.layers.Conv2D(64, (3, 3),
                                   strides=(2, 2), padding="same", activation=act)(x)
        x = tf.keras.layers.Conv2D(128, (3, 3), strides=(2, 2), padding="same", activation=act)(x)
        x = tf.keras.layers.Conv2D(256, (3, 3), strides=(2, 2), activation=act)(x)
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

        x = tf.keras.layers.Dense(decoder_struc[l], activation="sigmoid")(x)
        outp = tf.keras.layers.Reshape((28, 28, 1))(x)
        super(Bernoulli_Dense_Decoder, self).__init__(self.inp, outp, name="Decoder")
        self.summary()


class Bernoulli_Conv_Decoder(tf.keras.Model):
    def __init__(self, latent_dim, act):
        self.inp = x = tf.keras.Input(shape=(latent_dim,))
        x = tf.keras.layers.Dense(128, activation=act)(x)
        x = tf.keras.layers.Dense(3 * 3 * 128, activation=act)(x)
        x = tf.keras.layers.Reshape((3, 3, 128))(x)
        x = tf.keras.layers.Conv2DTranspose(64, (3, 3), strides=(2, 2), activation=act)(x)
        x = tf.keras.layers.Conv2DTranspose(32, (3, 3), strides=(
            2, 2), activation=act, padding="same")(x)
        outp = tf.keras.layers.Conv2DTranspose(1, (3, 3), strides=(
            2, 2), activation='sigmoid', padding="same")(x)
        super(Bernoulli_Conv_Decoder, self).__init__(self.inp, outp, name="Decoder")
        self.summary()


class Bernoulli_ConvTime_Decoder(tf.keras.Model):
    def __init__(self, latent_dim, act):
        self.inp = x = tf.keras.Input(shape=(latent_dim,))
        x = tf.keras.layers.Dense(256, activation=act)(x)
        x = tf.keras.layers.Dense(3 * 3 * 256, activation=act)(x)
        x = tf.keras.layers.Reshape((3, 3, 256))(x)
        x = tf.keras.layers.Conv2DTranspose(128, (3, 3), strides=(2, 2), activation=act)(x)
        x = tf.keras.layers.Conv2DTranspose(64, (3, 3), strides=(
            2, 2), activation=act, padding="same")(x)
        outp = tf.keras.layers.Conv2DTranspose(10, (3, 3), strides=(
            2, 2), activation='sigmoid', padding="same")(x)
        super(Bernoulli_ConvTime_Decoder, self).__init__(self.inp, outp, name="Decoder")
        self.summary()

# To Do:
# 1)
# Implementierung eines Gauß_Dense_Decoder, zum modellieren stetiger Daten.

# Dieser hat zwei Outputlayer, einen Erwartungswertoutput und ein log_var-Output,
# ähnlich wie im Encoder, nur ohne Reparametrisierung. Es muss außerdem eine
# neue Lossfunktion geschrieben werden, da log_p_xz anders berechnet wird.

# 2)
# Implementierung von  VAE_Conv_Encoder, Bernoulli_Conv_Decoder, Gauß_Conv_Decoder
# für Datensätze komplexer als MNIST. Überlegung einer geeigneten Modellstruktur
