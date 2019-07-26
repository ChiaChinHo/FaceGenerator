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


def ACGAN(path=None, modelname=None, z_dim=100, c_dim=1, lambda_c=1e-1):
    print('ACGAN')
    width=64
    height=64
    z_input = Input(shape = (z_dim, ))
    c_input = Input(shape = (c_dim, ))
    img_input = Input(shape = (width, height, 3))

    latent_dim = z_dim + c_dim
    nc = 32
    
    # concatenate
    latent = concatenate([z_input, c_input])
    latent_dim=z_dim+c_dim


    generator = Sequential([
                    Dense(2*2*latent_dim, input_shape=(latent_dim, )),
                    Reshape((2, 2, latent_dim)),
                    
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
                    Conv2D(nc, (5, 5), strides=2, input_shape=(width, height, 3), 
                        padding='same', name='d_conv1'),
                    LeakyReLU(alpha=0.2),

                    Conv2D(nc*1, (5, 5), strides=2, padding='same', name='d_conv2'),
                    BatchNormalization(),
                    LeakyReLU(alpha=0.2),

                    Conv2D(nc*2, (5, 5), strides=2, padding='same', name='d_conv3'),
                    LeakyReLU(alpha=0.2),

                    Conv2D(nc*2, (5, 5), strides=2, padding='same', name='d_conv4'),
                    BatchNormalization(),
                    LeakyReLU(alpha=0.2),

                    Conv2D(nc*4, (5, 5), strides=2, padding='same', name='d_conv5'),
                    BatchNormalization(),
                    LeakyReLU(alpha=0.2),
                    
                    Flatten()])

    feature = discriminator(img_input)
    fake = Dense(1, activation='sigmoid', name='Generation')(feature)
    aux = Dense(c_dim, activation='sigmoid', name='Auxiliary')(feature)

    def c_loss(y_true, y_pred):
        return lambda_c*binary_crossentropy(y_true, y_pred)

    g_opt = Adam(1e-4, 0.5)
    d_opt = Adam(1e-4, 0.5)
    losses = ['binary_crossentropy', c_loss]

    Gnet = Model(inputs=[z_input, c_input], outputs=generator(latent))
    Dnet = Model(inputs=img_input, outputs=[fake, aux])
    Gnet.compile(loss='binary_crossentropy', optimizer='SGD', metrics=['acc'])
    Dnet.compile(loss=losses, optimizer=d_opt, metrics=['acc'])


    Dnet.trainable = False
    GDnet = Model(inputs=[z_input, c_input], outputs=Dnet(Gnet([z_input, c_input])))
    if path and modelname:
        GDnet.load_weights(os.path.join(path, modelname))
    GDnet.compile(loss=losses, optimizer=g_opt, metrics=['acc'])

    #plot_model(GDnet, to_file='ACGAN.png')

    return GDnet, Gnet, Dnet

if __name__ == "__main__":
    ACGAN()
