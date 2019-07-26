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


def VAE(path=None, modelname=None, width=64, height=64, lambda_KL = 1e-5):
    print('VAE')
    nc=32
    def sampling(args):
        mu, log_var = args
        epsilon = K.random_normal(shape=(K.shape(mu)[0], latent_dim), 
            mean=0.0, stddev=1.0)
        return Add()([mu, K.exp(0.5*log_var) * epsilon])

    def mse(y_true, y_pred):
        return K.mean(K.square(y_true-y_pred)) 

    img_input = Input(shape = (width, height, 3))
    z = Input(shape=(latent_dim,))
    
    x = Conv2D(nc, (5, 5), strides=2, activation='relu', padding='same', name='conv1')(img_input)   # 32*32
    x = Conv2D(nc, (5, 5), strides=2, activation='relu', padding='same', name='conv2')(x)           # 16*16
    x = Conv2D(nc*2, (5, 5), strides=2, activation='relu', padding='same', name='conv3')(x)         # 8*8
    x = Conv2D(nc*2, (5, 5), strides=2, activation='relu', padding='same', name='conv4')(x)         # 4*4
    x = Conv2D(nc*4, (5, 5), strides=2, activation='relu', padding='same', name='conv5')(x)         # 2*2

    e_shape = Model(img_input, x).output_shape

    x = Flatten()(x)
    h_shape = Model(img_input, x).output_shape

    z_mean = Dense(latent_dim)(x)
    z_log_var = Dense(latent_dim)(x)
    z_sampling = Lambda(sampling)([z_mean, z_log_var])


    decode = Sequential([
                    Dense(h_shape[-1], input_shape=(latent_dim,)),
                    Activation('relu'),
                    Reshape(e_shape[1:]),

                    Conv2DTranspose(nc*4, (5, 5), strides=2, activation='relu', padding='same', name='dconv2'),
                    Conv2DTranspose(nc*2, (5, 5), strides=2, activation='relu', padding='same', name='dconv3'),
                    Conv2DTranspose(nc*2, (5, 5), strides=2, activation='relu', padding='same', name='dconv4'),
                    Conv2DTranspose(nc, (5, 5), strides=2, activation='relu', padding='same', name='dconv5'),
                    Conv2DTranspose(nc, (5, 5), strides=2, activation='relu', padding='same', name='dconv6'),
                    Conv2D(3, (5, 5), activation='tanh', padding='same', name='output')])

    def losses(y_true, y_pred):
        reconstruction_loss = K.mean(K.square(y_true-y_pred)) 
        KL_loss = -0.5* K.sum((1 + z_log_var - K.square(z_mean) - K.exp(z_log_var)), axis=-1)

        return K.mean(reconstruction_loss + lambda_KL * KL_loss)

    def MSE(y_true, y_pred):
        return K.mean(K.square(y_true-y_pred)) 

    x = decode(z_sampling)
    model = Model(img_input, x)

    opt = Adam(lr=1e-3)
    model.compile(loss=losses, optimizer=opt, metrics=[MSE])
    
    if path and modelname:
        print(os.path.join(path, modelname))
        model.load_weights(os.path.join(path, modelname))
    
    test_model = Model(inputs=img_input, outputs=[z_mean, decode(z_mean)])
    generator = Model(z, decode(z))

    #plot_model(model, to_file='VAE.png')

    return model, test_model, generator


if __name__ == "__main__":
    VAE()
