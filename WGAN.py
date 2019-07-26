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

def WGAN(path=None, modelname=None, z_dim=512, output_dim=1):
    print('WGAN')
    width=64
    height=64

    img_input = Input(shape = (width, height, 3))
    z = Input(shape=(z_dim, ))
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
                    
                    Conv2D(1, (5, 5), strides=2, padding='same', name='output'),
                    Flatten()])
        
    g_opt = RMSprop(lr=1e-4)
    d_opt = RMSprop(lr=1e-4)

    def W_loss(y_true, y_pred):
        return K.mean(y_true*y_pred)

    def acc(y_true, y_pred):
        return metrics.binary_accuracy(0.5*(-1*y_true+1), K.sigmoid(y_pred))

    Gnet = Model(z, generator(z))
    Gnet.compile(loss='binary_crossentropy', optimizer='SGD')

    Dnet = Model(img_input, discriminator(img_input))
    Dnet.compile(loss=W_loss, optimizer=d_opt, metrics=[acc])

    Dnet.trainable = False
    GDnet = Model(z, Dnet(Gnet(z)))
    if path and modelname:
        GDnet.load_weights(os.path.join(path, modelname))
    GDnet.compile(loss=W_loss, optimizer=g_opt, metrics=[acc])
    
    #plot_model(GDnet, to_file='WGAN.png')

    return GDnet, Gnet, Dnet


if __name__ == "__main__":
    WGAN()
