import tensorflow as tf
from keras import backend as K


def Bernoulli_Loss(encoder, decoder):
    μ, log_σ, z = encoder(encoder.inp)
    decoder_output = decoder(z)
    log_p_xz = K.sum(encoder.inp * K.log(decoder_output) +
                     (1. - encoder.inp) * K.log(1. - decoder_output), axis=-1)
    kl_div = .5 * K.sum(1. + 2. * μ - K.square(μ) - 2. * K.exp(log_σ), axis=-1)
    return (- log_p_xz - kl_div)
