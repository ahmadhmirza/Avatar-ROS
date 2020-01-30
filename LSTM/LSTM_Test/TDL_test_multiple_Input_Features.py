#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on Thu Jan 30 13:08:03 2020

@author: ahmad
"""


from numpy import array
import numpy as np
from keras.models import Sequential
from keras.layers import Dense
from keras.layers import LSTM
# prepare sequence
length = 5
seq = array([
        [1,2,3,4],
        [4,3,2,1],
        [3,4,5,6],
        [9,8,7,6],
        [1,3,5,7]
        ])

result = array([5,0,7,5,9])

#The input to LSTM must be 3-d
# To reshape seq to 3-d



# 5 samples, 4 timestep, 1 feature
x = seq.reshape(5, 4, 1)
# Output, 5 samples, 1 feature
y = result.reshape(5, 1)

# Network model:
# 1 input, 1 timestep,
# LSTM Layer: 5 units
# Output Layer: Fully Connected Layer - 1 output
# Fit using ADAM optimization algo, and mean squared error loss function
# batch size = number of samples:
### Avoids having to make LSTM stateful and manage state resets

# define LSTM configuration
n_neurons = length
n_batch = length
n_epoch = 1000
# create LSTM
model = Sequential()
model.add(LSTM(n_neurons, input_shape=(x.shape[1], x.shape[2])))
model.add(Dense(1))
model.compile(loss='mean_squared_error', optimizer='adam')
print(model.summary())
# train LSTM
model.fit(x, y, epochs=n_epoch, batch_size=n_batch, verbose=2)
# evaluate
result = model.predict(x, batch_size=n_batch, verbose=0)
for value in result:
	print('%.1f' % value)