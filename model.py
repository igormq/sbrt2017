from __future__ import absolute_import
from __future__ import division
from __future__ import print_function

import tensorflow as tf
import keras.backend as K
from keras.models import Model
from keras.layers import Input
from keras.layers import TimeDistributed
from keras.layers import Dense
from keras.layers import LSTM
from keras.layers import Bidirectional
from keras.layers import Lambda
from keras.layers import Dropout
from keras.regularizers import l2

def ler(y_true, y_pred, **kwargs):
    """
        Label Error Rate. For more information see 'tf.edit_distance'
    """
    return tf.reduce_mean(tf.edit_distance(y_pred, y_true, **kwargs))

def decode(inputs, **kwargs):
    """ Decodes a sequence of probabilities choosing the path with highest
    probability of occur

    # Arguments
        is_greedy: if True (default) the greedy decoder will be used;
        otherwise beam search decoder will be used

        if is_greedy is False:
            see the documentation of tf.nn.ctc_beam_search_decoder for more
            options

    # Inputs
        A tuple (y_pred, seq_len) where:
            y_pred is a tensor (N, T, C) where N is the bath size, T is the
            maximum timestep and C is the number of classes (including the
            blank label)
            seq_len is a tensor (N,) that indicates the real number of
            timesteps of each sequence

    # Outputs
        A sparse tensor with the top path decoded sequence

    """

    # Little hack for load_model
    import tensorflow as tf
    is_greedy = kwargs.get('is_greedy', True)
    y_pred, seq_len = inputs

    seq_len = tf.cast(seq_len[:, 0], tf.int32)
    y_pred = tf.transpose(y_pred, perm=[1, 0, 2])

    if is_greedy:
        decoded = tf.nn.ctc_greedy_decoder(y_pred, seq_len)[0][0]
    else:
        beam_width = kwargs.get('beam_width', 400)
        top_paths = kwargs.get('top_paths', 1)
        merge_repeated = kwargs.get('merge_repeated', True)

        decoded = tf.nn.ctc_beam_search_decoder(y_pred, seq_len, beam_width,
                                                top_paths,
                                                merge_repeated)[0][0]

    return decoded


def decode_output_shape(inputs_shape):
    y_pred_shape, seq_len_shape = inputs_shape
    return (y_pred_shape[:1], None)


def ctc_lambda_func(args):
    """ CTC cost function
    """
    y_pred, labels, inputs_length = args

    # Little hack for load_model
    import tensorflow as tf

    return tf.nn.ctc_loss(labels,
                          tf.transpose(y_pred, perm=[1, 0, 2]),
                          inputs_length[:, 0])


def ctc_dummy_loss(y_true, y_pred):
    """ Little hack to make CTC working with Keras
    """
    return y_pred


def decoder_dummy_loss(y_true, y_pred):
    """ Little hack to make CTC working with Keras
    """
    return K.zeros((1,))


def sbrt2017(num_hiddens, var_dropout, dropout, weight_decay, num_features=39,
             num_classes=28):
    """ SBRT model
    Reference:
        [1] Gal, Y, "A Theoretically Grounded Application of Dropout in
        Recurrent Neural Networks", 2015.
        [2] Graves, Alex, Abdel-rahman Mohamed, and Geoffrey Hinton. "Speech
        recognition with deep recurrent neural networks", 2013.
        [6] Wu, Yonghui, et al. "Google's Neural Machine Translation System:
        Bridging the Gap between Human and Machine Translation.", 2016.
    """

    x = Input(name='inputs', shape=(None, num_features))
    o = x

    if dropout > 0.0:
        o = Dropout(dropout)(o)

    o = Bidirectional(LSTM(num_hiddens,
                           return_sequences=True,
                           W_regularizer=l2(weight_decay),
                           U_regularizer=l2(weight_decay),
                           dropout_W=var_dropout,
                           dropout_U=var_dropout,
                           consume_less='gpu'))(o)

    if dropout > 0.0:
        o = Dropout(dropout)(o)

    o = TimeDistributed(Dense(num_classes,
                              W_regularizer=l2(weight_decay)))(o)

    # Define placeholders
    labels = Input(name='labels', shape=(None,), dtype='int32', sparse=True)
    inputs_length = Input(name='inputs_length', shape=(None,), dtype='int32')

    # Define a decoder
    dec = Lambda(decode, output_shape=decode_output_shape,
                 arguments={'is_greedy': True}, name='decoder')
    y_pred = dec([o, inputs_length])

    ctc = Lambda(ctc_lambda_func, output_shape=(1,), name="ctc")
    # Define loss as a layer
    loss = ctc([o, labels, inputs_length])

    return Model(input=[x, labels, inputs_length], output=[loss, y_pred])
