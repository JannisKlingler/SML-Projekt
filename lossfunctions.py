import tensorflow as tf
from keras import backend as K


def Bernoulli_Loss(encoder, decoder):
    decoder_output = decoder(encoder(encoder.inp)[2])
    log_p_xz = K.sum(encoder.inp * K.log(decoder_output) +
                     (1. - encoder.inp) * K.log(1. - decoder_output), axis=-1)
    kl_div = .5 * K.sum(1. + 2. * encoder(encoder.inp)[1] - K.square(encoder(encoder.inp)[0]) - 2. * K.exp(encoder(encoder.inp)[1]), axis=-1)
    return(- log_p_xz - kl_div)
