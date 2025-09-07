# -*- coding: utf-8 -*-

# from __future__ import print_function, division  # Ensures compatibility with Python 2/3 (for print and division behavior)

import numpy as np

# Keras modules for building the LSTM model
from keras.models import Sequential
from keras.optimizers import RMSprop
from keras.layers import Dense, LSTM
from keras.regularizers import l1
from keras.callbacks import EarlyStopping


class CustomLSTM:
    def __init__(self, num_hidden, num_channel, weight_decay=0.0):
        """
        Initialize the CustomLSTM model.

        Args:
            num_hidden (int): Number of hidden LSTM units.
            num_channel (int): Number of input features (channels).
            weight_decay (float): L1 regularization strength for weights.
        """

        # Create a sequential Keras model
        self.model = Sequential()

        # Add LSTM layer with L1 regularization
        self.model.add(
            LSTM(units=num_hidden,  # Number of hidden units in the LSTM layer
                input_shape=(None, num_channel),  # Variable-length sequences with 'num_channel' features
                kernel_regularizer=l1(weight_decay),  # L1 regularization on input weights
                recurrent_regularizer=l1(weight_decay)  # L1 regularization on recurrent weights
            )
        ) 
        self.model.add(Dense(1))

        # Print model summary to console
        self.model.summary()

        # Define the RMSProp optimizer with common hyperparameters
        rms_prop = RMSprop(learning_rate=0.001, rho=0.9, epsilon=1e-6)

        # Compile the model with mean squared error loss (common for regression)
        self.model.compile(loss='mean_squared_error', optimizer=rms_prop)

    def fit(self, x, y, batch_size=10, epochs=100):
        """
        Train the model on provided data.

        Args:
            x (ndarray): Input data of shape (samples, timesteps, features).
            y (ndarray): Target values.
            batch_size (int): Number of samples per gradient update.
            epochs (int): Maximum number of epochs to train.

        Returns:
            History object containing training metrics.
        """
        # Stop training early if validation loss doesn't improve for 5 consecutive epochs
        early_stopping = EarlyStopping(monitor='val_loss', patience=5, restore_best_weights=True)

        # Train the model with validation split and early stopping
        hist = self.model.fit(
            x, y,
            batch_size=batch_size,
            epochs=epochs,
            verbose=2,  # Print training output per epoch
            validation_split=0.2,  # Use 20% of training data as validation
            callbacks=[early_stopping]  # Stop early if overfitting
        )
        return hist

    def predict(self, x):
        return self.model.predict(x)
