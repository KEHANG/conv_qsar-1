from __future__ import print_function
from conv_qsar.utils.saving import save_model_history, save_model_history_manual
from conv_qsar.utils.GraphEmbedding import GraphFP
from conv_qsar.utils.neural_fp import sizeAttributeVector
from keras.models import Sequential
from keras.layers.core import Dense, Dropout
from keras.callbacks import LearningRateScheduler, EarlyStopping
from keras.optimizers import *
# from keras.utils.visualize_util import plot
import numpy as np
import datetime
import json
import sys
import os
from tqdm import tqdm


import keras.backend as K
import theano.tensor as T
K.is_nan = T.isnan 
K.logical_not = lambda x: 1 - x
def mse_no_NaN(y_true, y_pred):
	'''For each sample, sum squared error ignoring NaN values'''
	return K.sum(K.square(K.switch(K.logical_not(K.is_nan(y_true)), y_true, y_pred) - y_pred), axis = -1)

def binary_crossnetropy_no_NaN(y_true, y_pred):
	return K.sum(K.binary_crossentropy(K.switch(K.is_nan(y_true), y_pred, y_true), y_pred), axis = -1)

def build_model(embedding_size = 512, lr = 0.01, optimizer = 'adam', depth = 2, 
	scale_output = 0.05, padding = True, hidden = 0, hidden2 = 0, loss = 'mse', hidden_activation = 'tanh',
	output_activation = 'linear', dr1 = 0.0, dr2 = 0.0, output_size = 1, sum_after = False,
	molecular_attributes = False, use_fp = None):
	'''Generates simple embedding model to use molecular tensor as
	input in order to predict a single-valued output (i.e., yield)

	inputs:
		embedding_size - size of fingerprint for GraphFP layer
		lr - learning rate to use (train_model overwrites this value)
		optimizer - optimization function to use
		depth - depth of the neural fingerprint (i.e., radius)
		scale_output - initial scale for output weights in GraphFP
		padding - whether or not molecular tensors will be padded (i.e., batch_size > 1)
		hidden - number of hidden tanh nodes after FP (0 is linear)
		hidden2 - number of hidden nodes after "hidden" layer
		hidden_activation - activation function used in hidden layers
		output_activation - activation function for final output nodes
		dr1 - dropout rate after embedding
		dr2 - dropout rate after hidden
		loss - loss function as a string (e.g., 'mse')
		sum_after - whether to sum neighbor contributions after passing
					through a single network layer, or to do so before
					passing them from the network layer (during updates)
		molecular_attributes - whether to include additional molecular 
					attributes in the atom-level features (recommended)
		use_fp - whether the representation used is actually a fingerprint
					and not a convolutional network (for benchmarking)

	outputs:
		model - a Keras model'''

	# Base model
	model = Sequential()

	# Are we summing before or after? Two different defined "GraphFP" layers
	if sum_after:
		print('Using GraphFP with attribute summing *after* activation')
		from conv_qsar.utils.GraphEmbedding_sumAfter import GraphFP
	else:
		from conv_qsar.utils.GraphEmbedding import GraphFP

	# Are we using a convolutional embedding or a fingerprint representation?
	if type(use_fp) == type(None): # normal mode, use convolution
		model.add(GraphFP(embedding_size, sizeAttributeVector(molecular_attributes) - 1, 
			depth = depth,
			scale_output = scale_output,
			padding = padding,
			activation_inner = 'tanh'))
		print('    model: added GraphFP layer ({} -> {})'.format('mol', embedding_size))

	if hidden > 0:
		if dr1 > 0:
			model.add(Dropout(dr1))
			print('    model: Added Dropout({})'.format(dr1))
		if type(use_fp) == type(None): # convolutional model can inherit previous layer size
			model.add(Dense(hidden, activation = hidden_activation))
		else: # need to explicitly define fingerprint length since this is the first layer
			model.add(Dense(hidden, activation = hidden_activation, input_dim = 512))
		print('    model: added {} Dense layer (-> {})'.format(hidden_activation, hidden))
		if dr2 > 0:
			model.add(Dropout(dr2))
			print('    model: added Dropout({})'.format(dr2))
	if hidden2 > 0:
		model.add(Dense(hidden2, activation = hidden_activation))
		print('    model: added {} Dense layer (-> {})'.format(hidden_activation, hidden2))

	model.add(Dense(output_size, activation = output_activation))
	print('    model: added output Dense layer (-> {})'.format(output_size))

	# Compile
	if optimizer == 'adam':
		optimizer = Adam(lr = lr)
	elif optimizer == 'rmsprop':
		optimizer = RMSprop(lr = lr)
	elif optimizer == 'adagrad':
		optimizer = Adagrad(lr = lr)
	elif optimizer == 'adadelta':
		optimizer = Adadelta()
	else:
		print('Unrecognized optimizer')
		quit(1)

	# Custom loss to filter out NaN values in multi-task predictions
	if loss == 'custom':
		loss = mse_no_NaN
	elif loss == 'custom2':
		loss = binary_crossnetropy_no_NaN

	print('compiling...',)
	model.compile(loss = loss, optimizer = optimizer)
	print('done')

	return model

def save_model(model, loss, val_loss, fpath = '', config = {}, tstamp = ''):
	'''Saves NN model object and associated information.

	inputs:
		model - a Keras model
		loss - list of training losses 
		val_loss - list of validation losses
		fpath - root filepath to save everything to (with .json, h5, png, info)
		config - the configuration dictionary that defined this model 
		tstamp - current timestamp to log in info file'''

	# Dump data
	with open(fpath + '.json', 'w') as structure_fpath:
		json.dump(model.to_json(), structure_fpath)
	print('...saved structural information')

	# Dump weights
	model.save_weights(fpath + '.h5', overwrite = True)
	print('...saved weights')

	# # Dump image
	# plot(model, to_file = fpath + '.png')
	# print('...saved image')

	# Dump history
	save_model_history_manual(loss, val_loss, fpath + '.hist')
	print ('...saved history')

	# Write to info file
	info_fid = open(fpath + '.info', 'a')
	info_fid.write('{} saved {}\n\n'.format(fpath, tstamp))
	info_fid.write('Configuration details\n------------\n')
	info_fid.write('  {}\n'.format(config))
	info_fid.close()

	print('...saved model to {}.[json, h5, png, info]'.format(fpath))
	return True


def train_model(model, data, nb_epoch = 0, batch_size = 1, lr_func = None, patience = 10):
	'''Trains the model.

	inputs:
		model - a Keras model
		data - three dictionaries for training,
				validation, and testing separately
		nb_epoch - number of epochs to train for
		batch_size - batch_size to use on the data. This must agree with what was
				specified for data (i.e., if tensors are padded or not)
		lr_func - string which is evaluated with 'epoch' to produce the learning 
				rate at each epoch 
		patience - number of epochs to wait when no progress is being made in 
				the validation loss. a patience of -1 means that the model will
				use weights from the best-performing model during training

	outputs:
		model - a trained Keras model
		loss - list of training losses corresponding to each epoch 
		val_loss - list of validation losses corresponding to each epoch'''

	# Unpack data 
	(train, val, test) = data
	mols_train = train['mols']; y_train = train['y']; smiles_train = train['smiles']
	mols_val   = val['mols'];   y_val   = val['y'];   smiles_val   = val['smiles']
	print('{} to train on'.format(len(mols_train)))
	print('{} to validate on'.format(len(mols_val)))
	print('{} to test on'.format(len(smiles_val)))

	# Create learning rate function
	if lr_func:
		lr_func_string = 'def lr(epoch):\n    return {}\n'.format(lr_func)
		exec lr_func_string

	# Fit (allows keyboard interrupts in the middle)
	# Because molecular graph tensors are different sizes based on N_atoms, can only do one at a time
	# (alternative is to pad with zeros and try to add some masking feature to GraphFP)
	# -> this is why batch_size == 1 is treated distinctly
	try:
		loss = []
		val_loss = []

		if batch_size == 1: # DO NOT NEED TO PAD
			wait = 0
			prev_best_val_loss = 99999999
			for i in range(nb_epoch):
				this_loss = []
				this_val_loss = []
				if lr_func: model.optimizer.lr.set_value(lr(i))
				print('Epoch {}/{}, lr = {}'.format(i + 1, nb_epoch, model.optimizer.lr.get_value()))

				# Run through training set
				print('Training...')
				training_order = range(len(mols_train))
				np.random.shuffle(training_order)
				for j in tqdm(training_order):
					single_mol_as_array = np.array(mols_train[j:j+1])

					single_y_as_array = np.reshape(y_train[j], (1, -1))
					sloss = model.train_on_batch(single_mol_as_array, single_y_as_array)
					this_loss.append(sloss)

				# Run through testing set
				print('Validating..')
				for j in tqdm(range(len(mols_val))):
					single_mol_as_array = np.array(mols_val[j:j+1])
					single_y_as_array = np.reshape(y_val[j], (1, -1))
					sloss = model.test_on_batch(single_mol_as_array, single_y_as_array)
					this_val_loss.append(sloss)
				
				loss.append(np.mean(this_loss))
				val_loss.append(np.mean(this_val_loss))
				print('loss: {}\tval_loss: {}'.format(loss[i], val_loss[i]))

				# Check progress
				if np.mean(this_val_loss) < prev_best_val_loss:
					wait = 0
					prev_best_val_loss = np.mean(this_val_loss)
					if patience == -1:
						model.save_weights('best.h5', overwrite=True)
				else:
					wait = wait + 1
					print('{} epochs without val_loss progress'.format(wait))
					if wait == patience:
						print('stopping early!')
						break
			if patience == -1:
				model.load_weights('best.h5')

		else: 
			# When the batch_size is larger than one, we have padded mol tensors
			# which  means we need to concatenate them but can use Keras' built-in
			# training functions with callbacks, validation_split, etc.
			if lr_func:
				callbacks = [LearningRateScheduler(lr)]
			else:
				callbacks = []
			if patience != -1:
				callbacks.append(EarlyStopping(patience = patience, verbose = 1))

			if mols_val:
				mols = np.vstack((mols_train, mols_val))
				y = np.concatenate((y_train, y_val))
				hist = model.fit(mols, y, 
					nb_epoch = nb_epoch, 
					batch_size = batch_size, 
					validation_split = (1 - float(len(mols_train))/(len(mols_val) + len(mols_train))),
					verbose = 1,
					callbacks = callbacks)	
			else:
				hist = model.fit(np.array(mols_train), np.array(y_train), 
					nb_epoch = nb_epoch, 
					batch_size = batch_size, 
					verbose = 1,
					callbacks = callbacks)	
			
			loss = []; val_loss = []
			if 'loss' in hist.history: loss = hist.history['loss']
			if 'val_loss' in hist.history: val_loss = hist.history['val_loss']

	except KeyboardInterrupt:
		print('User terminated training early (intentionally)')

	return (model, loss, val_loss)
