# -*- coding: utf-8 -*-
"""
@author: Imanne El Maachi
"""

import numpy as np
np.random.seed(2)

# from tensorflow.python.framework.ops import disable_eager_execution
# disable_eager_execution()

from tensorflow.keras import layers
from tensorflow.keras import Input
from tensorflow.keras import optimizers
from tensorflow.keras.models import Model
from tensorflow.keras.layers import Dropout, Flatten, Conv1D



def conv1D_full():
    '''

    :return: 1 branch of the parallel Convnet
    '''
    input1 = Input(shape=(100, 1))
    x = Conv1D(filters=8, kernel_size=3, activation='selu', padding='valid')(input1)
    x = Conv1D(filters=16, kernel_size=3, activation='selu', padding='valid')(x)
    x = layers.MaxPooling1D(2)(x)
    x = Conv1D(filters=16, kernel_size=3, activation='selu', padding='valid')(x)
    x = Conv1D(filters=16, kernel_size=3, activation='selu', padding='valid')(x)
    x = layers.MaxPooling1D(2)(x)
    x = layers.Flatten()(x)
    x = layers.Dense(50, activation='elu')(x)
    model = Model(input1, x)
    # rms = optimizers.RMSprop(lr=0.001, decay=0.005)
    rms = optimizers.RMSprop(lr=0.001)
    model.compile(loss='categorical_crossentropy', optimizer=rms, metrics=['accuracy'], experimental_run_tf_function=False)
    print(model.summary())
    return input1, x


def multiple_cnn1D(nb):
    '''

    :param nb: number of features ( indicates the number of parallel branches)
    :return:
    '''
    # initialise with the first input
    inputs = []
    CNNs = []
    
    input_, CNN_ = conv1D_full() # iadd the first inputs to array
    inputs.append( input_ ) 
    CNNs.append( CNN_ )

    # inputs = np.array(input_)  # iadd the first inputs to array
    # CNNs = np.array(CNN_)


    for i in range(1,nb ):
        input_i, CNN_i = conv1D_full()
        inputs.append( input_i ) 
        CNNs.append( CNN_i )

        # inputs = np.append(inputs, input_i)
        # CNNs = np.append(CNNs, CNN_i)

    # concatenated = layers.add(LSTMs.tolist())
    x = layers.concatenate(CNNs, axis=-1)
    x = Dropout(0.5)(x)
    x = layers.Dense(100, activation='selu')(x)
    x =  Dropout(0.5)(x)
    x = layers.Dense(20, activation='selu')(x)
    x = Dropout(0.5)(x)
    answer = layers.Dense(1, activation='sigmoid')(x)
    model = Model(inputs, answer)
    opt = optimizers.RMSprop(lr=0.001)
    model.compile(loss='binary_crossentropy', optimizer=opt, metrics=['accuracy'], experimental_run_tf_function=False)
    print(model.summary())
    return model


def multiple_cnn1D5_level(nb):
    '''
    Model for severity prediction , 5 classes output
    :param nb:  number of parallel branch
    :return:
    '''
    # initialise with the first input
    input_, CNN_ = conv1D_full()
    inputs = np.array(input_)  # iadd the first inputs to array
    CNNs = np.array(CNN_)

    for i in range(1,nb ):
        input_i, CNN_i = conv1D_full()
        inputs = np.append(inputs, input_i)
        CNNs = np.append(CNNs, CNN_i)

    x = layers.concatenate(CNNs.tolist(), axis=-1)
    x = Dropout(0.5)(x)
    x = layers.Dense(100, activation='selu')(x)
    x =  Dropout(0.5)(x)
    x = layers.Dense(20, activation='selu')(x)
    x = Dropout(0.5)(x)
    answer = layers.Dense(5, activation='softmax')(x)
    model = Model(inputs.tolist(), answer)
    opt = optimizers.Nadam(lr=0.001)
    model.compile(loss='categorical_crossentropy', optimizer=opt, metrics=['accuracy'], experimental_run_tf_function=False)
    print(model.summary())
    return model
