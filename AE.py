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

def AE(path=None, width=64, height=64):
    img_input = Input(shape = (width, height, 3))

    x = Conv2D(64, (3, 3), activation='relu', padding='same', name='conv1')(img_input)
    x = MaxPooling2D((2, 2), strides=(2, 2), padding='same', name = 'pool1')(x)
    x = Conv2D(32, (3, 3), activation='relu', padding='same', name='conv2')(x)
    x = MaxPooling2D((2, 2), strides=(2, 2), padding='same', name = 'pool2')(x)
    x = Conv2D(16, (3, 3), activation='relu', padding='same', name='conv3')(x)
    x = MaxPooling2D((2, 2), strides=(2, 2), padding='same', name = 'pool3')(x)

    x = Conv2D(16, (3, 3), activation='relu', padding='same', name='conv4')(x)
    x = UpSampling2D((2, 2))(x)
    x = Conv2D(32, (3, 3), activation='relu', padding='same', name='conv5')(x)
    x = UpSampling2D((2, 2))(x)
    x = Conv2D(64, (3, 3), activation='relu', padding='same', name='conv6')(x)
    x = UpSampling2D((2, 2))(x)
    x = Conv2D(3, (3, 3), activation='tanh', padding='same', name='output')(x)

    model = Model(img_input, x)

    reconstruction_loss = K.mean(K.square(img_input-x)) 
    model.add_loss(reconstruction_loss)
    opt = Adam()
    model.compile(optimizer=opt)
    model.summary()
    return model

if __name__ == "__main__":
    AE()
