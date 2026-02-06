# -*- coding: utf-8 -*-

import numpy as np

# Keras modules for building the LSTM model
from keras.models import Sequential
from keras.optimizers import RMSprop
from keras.layers import Dense, LSTM
from keras.regularizers import l1
from keras.callbacks import EarlyStopping


class CustomLSTM:
    def __init__(self, num_hidden, num_channel, weight_decay=0.0):

        self.model = Sequential()
        self.model.add(
            LSTM(units=num_hidden,
                input_shape=(None, num_channel),
                kernel_regularizer=l1(weight_decay),    # L1 regularization on input weights
                recurrent_regularizer=l1(weight_decay)  # L1 regularization on recurrent weights
            ))
        self.model.add(Dense(1))
        self.model.summary()
        rms_prop = RMSprop(learning_rate=0.001, rho=0.9, epsilon=1e-6)
        self.model.compile(loss='mean_squared_error', optimizer=rms_prop)

    def fit(self, x, y, batch_size=10, epochs=100):

        # Stop training early if validation loss doesn't improve for 5 consecutive epochs
        early_stopping = EarlyStopping(monitor='val_loss', patience=5, restore_best_weights=True)

        # Train the model with validation split and early stopping
        hist = self.model.fit(
            x, y,
            batch_size=batch_size,
            epochs=epochs,
            verbose=1,                  # Print training output per epoch
            validation_split=0.2,       # Use 20% of training data as validation
            callbacks=[early_stopping]  # Stop early if overfitting
        )
        return hist

    def predict(self, x):
        return self.model.predict(x)
