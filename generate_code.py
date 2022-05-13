import sys
from tensorflow import keras
import numpy as np
from cnn_template import cnn_model
os.environ['TF_CPP_MIN_LOG_LEVEL'] = '2'


options = ['SLayerS', 'SConvTypeS', 'SOptimizerS', 'SPoolingFnS', 'SActivationFnS']

def generate_token(state):
    return "STK" + str(state) + "STK"

def abstraction(input_string):
	curr_mapping = {}
	inverse_map = {}
	state = 0
	text = input_string
	text = text.replace('.', ' ')
	text = text.replace(',', ' ')
	text = text.split(' ')
	new_text = []
	tmp = []
	# print(text)
	for i in text:
		if i.isnumeric():
			curr_mapping[i] = generate_token(state)
			inverse_map[generate_token(state)] = i
			state += 1
			new_text.append(curr_mapping[i])
		else:
			new_text.append(i)

	inp_string = ' '.join(new_text)
	return [inp_string, curr_mapping]

if __name__ == '__main__':
	
	# input_string = input('Enter the model specification in natural language: ')
	model = keras.models.load_model('task1', compile=False)

	sample_text = ("Generate a Convolutional Neural Network with {convolution_layer} type of layers. The count needs to be {layer_count}. Use {optimizer} for optimization and {pooling_function} as the pooling layer. Also use {activation_function} for activation with {n1} filters for {k1} layer. The {k2} layer has {n2} filters, {k3} layer has {n3} filters and {n4} filters for layer {k4}. The rest have {n5} filters."
	.format(layer_count=5, convolution_layer='2D separable convolution', activation_function='gelu', optimizer='nadam', pooling_function='average pooling', n1='second', n2='fifth', n3='first', n4='sixth', k1='2', k2='10', k3='15', k4='20', k5='1'))
	print(sample_text)
	# ("Generate a Convolutional Neural Network with {convolution_layer} type of layers. The count needs to be {layer_count}. Use {optimizer} for optimization and {pooling_function} as the pooling layer. Also use {activation_function} for activation."
	# ("Generate CNN code for classification with layer count of {layer_count} and use {convolution_layer} layers. Use {activation_function} as activation function and {optimizer} optimizer. The pooling layer is {pooling_function}."
	# Convolution Layer
	input_string = sample_text + " " + options[1]
	predictions = model.predict(np.array([input_string]))
	conv_layer = np.argmax(predictions[0])

	# Activation Function
	input_string = sample_text + " " + options[4]
	predictions = model.predict(np.array([input_string]))
	activation_fn = np.argmax(predictions[0]) - 13
	# print(predictions[0])
	# print(np.argmax(predictions[0]))

	# Pooling Function
	input_string = sample_text + " " + options[3]
	predictions = model.predict(np.array([input_string]))
	pooling_fn = np.argmax(predictions[0]) - 9

	# Optimizer
	input_string = sample_text + " " + options[2]
	predictions = model.predict(np.array([input_string]))
	optimizer = np.argmax(predictions[0]) - 25


	# print(layer_count)
	# print(conv_layer)
	# print(activation_fn)
	# print(pooling_fn)
	# print(optimizer)

	model = keras.models.load_model('task2', compile=False)
	tmp = abstraction(sample_text)
	input_string = tmp[0] + " S0S"
	var_map = tmp[1]
	predictions = model.predict(np.array([input_string]))
	layer_count = np.argmax(predictions[0])
	layer_count = int(var_map[layer_count])

	filters = []
	for i in range(1, layer_count + 1):
		input_string = tmp[0] + " S" + str(i) + "S"
		predictions = model.predict(np.array([input_string]))
		filter_count = np.argmax(predictions[0])
		filter_count = int(var_map[layer_count])
		filters.append(filter_count)

	cnn_model(layer_count, conv_layer, pooling_fn, activation_fn, optimizer, filters)