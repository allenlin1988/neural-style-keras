'''
This module contains functions for building the pastiche model.
'''

import keras
from keras.models import Model
from keras.layers import (Conv2D, Activation, UpSampling2D,
                          ZeroPadding2D, Input, BatchNormalization,
                          add, Lambda)
from layers import (ReflectionPadding2D, InstanceNormalization,
                    ConditionalInstanceNormalization)
from keras.initializers import random_normal
import keras.backend as K

# Initialize weights with normal distribution with std 0.01
def weights_init(shape, dtype=None):
    return K.random_normal(shape, 0, 0.01, dtype=dtype)

def conv(x, n_filters, kernel_size=3, stride=1, relu=True, nb_classes=1, targets=None):
    '''
    Reflection padding, convolution, instance normalization and (maybe) relu.
    '''
    if not kernel_size % 2:
        raise ValueError('Expected odd kernel size.')
    pad = (kernel_size - 1) / 2
    o = ReflectionPadding2D(padding=(pad, pad))(x)
    o = Conv2D(n_filters, (kernel_size, kernel_size), strides=(stride, stride), kernel_initializer=weights_init)(o)
    # o = BatchNormalization()(o)
    if nb_classes > 1:
        o = ConditionalInstanceNormalization(targets, nb_classes)(o)
    else:
        o = InstanceNormalization()(o)
    if relu:
        o = Activation('relu')(o)
    return o


def residual_block(x, n_filters, nb_classes=1, targets=None):
    '''
    Residual block with 2 3x3 convolutions blocks. Last one is linear (no ReLU).
    '''
    o = conv(x, n_filters)
    # Linear activation on second conv
    o = conv(o, n_filters, relu=False, nb_classes=nb_classes, targets=targets)
    # Shortcut connection
    o = add([o, x])
    return o


def upsampling(x, n_filters, nb_classes=1, targets=None):
    '''
    Upsampling block with nearest-neighbor interpolation and a conv block.
    '''
    o = UpSampling2D()(x)
    o = conv(o, n_filters, nb_classes=nb_classes, targets=targets)
    return o


def pastiche_model(img_size, width_factor=2, nb_classes=1, targets=None):
    k = width_factor
    x = Input(shape=(img_size, img_size, 3))
    o = conv(x, 16 * k, kernel_size=9, nb_classes=nb_classes, targets=targets)
    o = conv(o, 32 * k, stride=2, nb_classes=nb_classes, targets=targets)
    o = conv(o, 64 * k, stride=2, nb_classes=nb_classes, targets=targets)
    o = residual_block(o, 64 * k, nb_classes=nb_classes, targets=targets)
    o = residual_block(o, 64 * k, nb_classes=nb_classes, targets=targets)
    o = residual_block(o, 64 * k, nb_classes=nb_classes, targets=targets)
    o = residual_block(o, 64 * k, nb_classes=nb_classes, targets=targets)
    o = residual_block(o, 64 * k, nb_classes=nb_classes, targets=targets)
    o = upsampling(o,  32 * k, nb_classes=nb_classes, targets=targets)
    o = upsampling(o, 16 * k, nb_classes=nb_classes, targets=targets)
    o = conv(o, 3, kernel_size=9, relu=False, nb_classes=nb_classes, targets=targets)
    o = Activation('tanh')(o)
    o = Lambda(lambda x: 150*x, name='scaling')(o)
    pastiche_net = Model(inputs=[x], outputs=[o])
    return pastiche_net

"""
Study note: 
 
"""