import numpy as np
from os import path

from keras.models import Model
from keras.layers import merge, Convolution2D, MaxPooling2D, Input, Dense, Flatten, Dropout, Reshape, TimeDistributed, BatchNormalization, Merge, merge
from keras.layers.advanced_activations import LeakyReLU
from keras.layers import LSTM


class PortfolioDDPG:
	def __init__(self, weights_path = None):
		self.weights_path = weights_path

	def getModel(self):
		weights_path = self.weights_path
		model = self.buildModel()

		if weights_path and path.isfile(weights_path):
			try:
				model.load_weights(weights_path)
			except Exception as e:
				print(e)

		return model

	def buildModel(self):

		B = Input(shape = (5,))
		b = Dense(8, activation = "relu")(B)
		b = Dense(4, activation="relu")(b)
		# b = LSTM(8, activation="relu")(B)
		inputs = [B]
		merges = [b]

		for i in range(1):
			S = Input(shape=[5, 20 * 42 , 1])
			inputs.append(S)
			# h = Convolution2D(2048, 3, 1, border_mode = 'same')(S)
			# h = LeakyReLU(0.001)(h)
			h = Convolution2D(256, 40, 1, border_mode = 'same')(S)
			h = LeakyReLU(0.001)(h)
			# h = Convolution2D(512, 40, 1, border_mode = 'same')(S)
			# h = LeakyReLU(0.001)(h)
			# h = Convolution2D(2048, 20, 1, border_mode = 'same')(S)
			# h = LeakyReLU(0.001)(h)
			# h = Convolution2D(1024, 40, 1, border_mode = 'same')(S)
			# h = LeakyReLU(0.001)(h)

			h = Flatten()(h)
			h = Dense(128)(h)
			h = LeakyReLU(0.001)(h)
			h = Dense(64)(h)
			h = LeakyReLU(0.001)(h)
			merges.append(h)

			# h = Convolution2D(512, 60, 1, border_mode = 'same')(S)
			# h = LeakyReLU(0.001)(h)
    			# h = Flatten()(h)
			# h = Dense(128)(h)
			# h = LeakyReLU(0.001)(h)
			# merges.append(h)

		m = merge(merges, mode = 'concat', concat_axis = 1)
		# m = Dense(512)(m)
		# m = LeakyReLU(0.001)(m)
		m = Dense(64)(m)
		m = LeakyReLU(0.001)(m)
		m = Dense(32)(m)
		m = LeakyReLU(0.001)(m)
		V = Dense(5, activation = 'softmax')(m)
		model = Model(input = inputs, output = V)

		return model
