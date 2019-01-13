#!/usr/bin/env python
from __future__ import print_function

import skimage as skimage
from skimage import transform, color, exposure
from skimage.viewer import ImageViewer
import random
from random import choice
import numpy as np
from collections import deque
import time

import json
from keras.models import model_from_json
from keras.models import Sequential, load_model, Model
from keras.layers.core import Dense, Dropout, Activation, Flatten
from keras.layers.wrappers import TimeDistributed
from keras.layers import Convolution2D, Dense, Flatten, merge, MaxPooling2D, Input, AveragePooling2D, Lambda, Merge, Activation, Embedding
from keras.optimizers import SGD, Adam, rmsprop
from keras.layers.recurrent import LSTM, GRU
from keras.layers.normalization import BatchNormalization
from keras import backend as K

import tensorflow as tf
#tf.python.control_flow_ops = tf


class Average(Merge):
    """Layer that averages a list of inputs.
    It takes as input a list of tensors,
    all of the same shape, and returns
    a single tensor (also of the same shape).
    """

    def _merge_function(self, inputs):
        output = inputs[0]
        for i in range(1, len(inputs)):
            output += inputs[i]
        return output / len(inputs)

class Subtract(Merge):
    """Layer that subtracts two inputs.
    It takes as input a list of tensors of size 2,
    both of the same shape, and returns a single tensor, (inputs[0] - inputs[1]),
    also of the same shape.
    # Examples
    ```python
        import keras
        input1 = keras.layers.Input(shape=(16,))
        x1 = keras.layers.Dense(8, activation='relu')(input1)
        input2 = keras.layers.Input(shape=(32,))
        x2 = keras.layers.Dense(8, activation='relu')(input2)
        # Equivalent to subtracted = keras.layers.subtract([x1, x2])
        subtracted = keras.layers.Subtract()([x1, x2])
        out = keras.layers.Dense(4)(subtracted)
        model = keras.models.Model(inputs=[input1, input2], outputs=out)
    ```
    """

    def build(self, input_shape):
        super(Subtract, self).build(input_shape)
        if len(input_shape) != 2:
            raise ValueError('A `Subtract` layer should be called '
                             'on exactly 2 inputs')

    def _merge_function(self, inputs):
        if len(inputs) != 2:
            raise ValueError('A `Subtract` layer should be called '
                             'on exactly 2 inputs')
        return inputs[0] - inputs[1]



class Networks(object):

    @staticmethod
    def dfp_network(input_shape, measurement_size, goal_size, action_size, num_timesteps, learning_rate):
        """
        Neural Network for Direct Future Predition (DFP)
        """

        # Perception Feature
        state_input = Input(shape=(input_shape))
        perception_feat = Convolution2D(32, 8, 8, subsample=(4,4), activation='relu')(state_input)
        perception_feat = Convolution2D(64, 4, 4, subsample=(2,2), activation='relu')(perception_feat)
        perception_feat = Convolution2D(64, 3, 3, activation='relu')(perception_feat)
        perception_feat = Flatten()(perception_feat)
        perception_feat = Dense(512, activation='relu')(perception_feat)

        # Measurement Feature
        measurement_input = Input(shape=((measurement_size,)))
        measurement_feat = Dense(128, activation='relu')(measurement_input)
        measurement_feat = Dense(128, activation='relu')(measurement_feat)
        measurement_feat = Dense(128, activation='relu')(measurement_feat)

        # Goal Feature
        goal_input = Input(shape=((goal_size,)))
        goal_feat = Dense(128, activation='relu')(goal_input)
        goal_feat = Dense(128, activation='relu')(goal_feat)
        goal_feat = Dense(128, activation='relu')(goal_feat)

        concat_feat = merge([perception_feat, measurement_feat, goal_feat], mode='concat')

        measurement_pred_size = measurement_size * num_timesteps # 3 measurements, 6 timesteps

        expectation_stream = Dense(measurement_pred_size, activation='relu')(concat_feat)

        action_stream = []
        for i in range(action_size):
            action_stream.append(Dense(measurement_pred_size, activation='relu')(concat_feat))

        avg = Average()(action_stream)

        prediction_list = []
        for i in range(action_size):
            tmp = Subtract()([action_stream[i], avg])
            prediction_list.append(merge([tmp, expectation_stream], mode='sum'))

        model = Model(input=[state_input, measurement_input, goal_input], output=prediction_list)

        adam = Adam(lr=learning_rate)
        model.compile(loss='mse',optimizer=adam)

        return model
