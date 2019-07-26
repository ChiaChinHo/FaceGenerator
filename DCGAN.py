import os
import numpy as np
from keras import backend
from keras.models import Model, Sequential
from keras.layers import *
import keras.backend as K
from keras import metrics, regularizers
from keras.optimizers import *
from keras.losses import *
#from keras.utils.vis_utils import plot_model

latent_dim = 512

def DCGAN(path=None, modelname=None, z_dim=512, output_dim=2):
    print('DCGAN')
    width=64
    height=64

    img_input = Input(shape = (width, height, 3))
    z = Input(shape=(z_dim, ))

    filter_dim = 256
    nc = 32

    generator = Sequential([
                    Dense(2*2*z_dim, input_shape=(z_dim, )),
                    Reshape((2, 2, z_dim)),

                    Conv2DTranspose(nc*4, (5, 5), strides=2, padding='same', name='gconv1'),
                    LeakyReLU(alpha=0.2),
                    BatchNormalization(),

                    Conv2DTranspose(nc*2, (5, 5), strides=2, padding='same', name='gconv2'),
                    LeakyReLU(alpha=0.2),
                    BatchNormalization(),

                    Conv2DTranspose(nc*2, (5, 5), strides=2, padding='same', name='gconv3'),
                    LeakyReLU(alpha=0.2),
                    BatchNormalization(),

                    Conv2DTranspose(nc*1, (5, 5), strides=2, padding='same', name='gconv4'),
                    LeakyReLU(alpha=0.2),
                    BatchNormalization(),

                    Conv2DTranspose(nc*1, (5, 5), strides=2, padding='same', name='gconv5'),
                    LeakyReLU(alpha=0.2),
                    BatchNormalization(),

                    Conv2D(3, (5, 5), activation='tanh', padding='same', name='output')])

    discriminator = Sequential([
                    Conv2D(nc, (5, 5), strides=2, input_shape=(width, height, 3), padding='same', name='d_conv1'),
                    LeakyReLU(alpha=0.2),

                    Conv2D(nc, (5, 5), strides=2, padding='same', name='d_conv2'),
                    BatchNormalization(),
                    LeakyReLU(alpha=0.2),

                    Conv2D(nc*2, (5, 5), strides=2, padding='same', name='d_conv3'),
                    BatchNormalization(),
                    LeakyReLU(alpha=0.2),

                    Conv2D(nc*2, (5, 5), strides=2, padding='same', name='d_conv4'),
                    BatchNormalization(),
                    LeakyReLU(alpha=0.2),

                    Conv2D(nc*4, (5, 5), strides=2, padding='same', name='d_conv5'),
                    BatchNormalization(),
                    LeakyReLU(alpha=0.2),

                    Conv2D(1, (3, 3), strides=2, padding='same', name='output'),
                    Activation('sigmoid'),

                    Flatten()])

    
    g_opt = Adam(1e-4, 0.5)
    d_opt = Adam(1e-4, 0.5)

    Gnet = Model(z, generator(z))
    Gnet.compile(loss='binary_crossentropy', optimizer="SGD")

    Dnet = Model(img_input, discriminator(img_input))
    Dnet.compile(loss='binary_crossentropy', optimizer=d_opt, metrics=['acc'])

    Dnet.trainable = False
    GDnet = Model(z, Dnet(Gnet(z)))
    if path and modelname:
        GDnet.load_weights(os.path.join(path, modelname))
    GDnet.compile(loss='binary_crossentropy', optimizer=g_opt, metrics=['acc'])
    Dnet.trainable = True
    

    return GDnet, Gnet, Dnet


if __name__ == "__main__":
    DCGAN()
