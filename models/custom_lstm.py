# -*- coding: utf-8 -*-

from __future__ import print_function, division

import numpy as np

from keras.models import Sequential
from keras.optimizers import RMSprop
from keras.layers import Dense, LSTM
from keras.regularizers import l1
from keras.callbacks import EarlyStopping


class CustomLSTM:
    def __init__(self, num_hidden=10, num_channel=5, weight_decay=0.0):
        if num_hidden is None:
            num_hidden = num_channel

        self.model = Sequential()

        self.model.add(LSTM(
            units=num_hidden,  
            input_shape=(None, num_channel),  # (timesteps, features)
            kernel_regularizer=l1(weight_decay),
            recurrent_regularizer=l1(weight_decay)
        ))    

        self.model.add(Dense(1))

        self.model.summary()

        rms_prop = RMSprop(learning_rate=0.001, rho=0.9, epsilon=1e-6)

        self.model.compile(loss='mean_squared_error', optimizer=rms_prop)

    def fit(self, x, y, batch_size=10, epochs=100):
        early_stopping = EarlyStopping(monitor='val_loss', patience=5, restore_best_weights=True)

        hist = self.model.fit(
            x, y,
            batch_size=batch_size,
            epochs=epochs,
            verbose=2,
            validation_split=0.2,
            callbacks=[early_stopping]
        )
        return hist

    def predict(self, x):
        return self.model.predict(x)