import csv
import random
import pandas as pd

SAMPLES_COUNT = 500000
PERCENT_TRAIN = 0.8

templates = []
convolution_layers = []
activation_functions = []
pooling_functions = []
optimizers = []
max_layer_count = 10
max_filter_count = 100

templates = [
"Write a CNN model with {layer_count} {convolution_layer} layers and {pooling_function}. The activation to be used is {activation_function} and optimizer is {optimizer}. The number of filters in each layer is {n1}.",
"Synthesize CNN code for classification. The model should have {convolution_layer} layers of count {layer_count}. The {k1}, {k2}, {k3} layers have {n1}, {n2}, {n3} filters and the rest have {n4}. {activation_function} is to be used for activation. The pooling function is {pooling_function}. The model should use {optimizer} optimizer.",
"A CNN model using {optimizer} optimizer needs to be used with {layer_count} {convolution_layer} layers. The activation function and pooling layer are {activation_function} and {pooling_function} respectively. The {k1} layer has {n1} filters. The rest have {n2} filters.",
"CNN code for classification needs to be synthesized. The optimizer is {optimizer}, activation is {activation_function} and pooling layer is {pooling_function}. The number of layers is {layer_count} with {convolution_layer} and with {k1}, {k2}, {k3}, {k4} layers having {n1}, {n2}, {n3}, {n4} filters and the rest having {n5}.",
"Generate CNN code for classification with layer count of {layer_count} and use {convolution_layer} layers. Use {activation_function} as activation function and {optimizer} optimizer. The pooling layer is {pooling_function} and use {n1} filters for {k1} layer, {n2} filters for {k2} layer and {n3} filters for the rest.",
"Generate a Convolutional Neural Network with {convolution_layer} type of layers. The count needs to be {layer_count}. Use {optimizer} for optimization and {pooling_function} as the pooling layer. Also use {activation_function} for activation with {n1} filters for {k1} layer. The {k2} layer has {n2} filters, {k3} layer has {n3} filters and {n4} filters for layer {k4}. The rest have {n5} filters.",
"A Convolutional Neural Network with {layer_count} {pooling_function} for pooling and {convolution_layer} layers. To use {activation_function} for the layers and {optimizer} for optimization. The number of filters in each layer is {n1}.",
"For classification, synthesize code in tensorflow with {convolution_layer} layers of count {layer_count}. The pooling layer is {pooling_function} and activation used is {activation_function} along with {optimizer} optimizer. The number of layers is {layer_count} with {convolution_layer} and with {k1}, {k2}, {k3}, {k4} layers having {n1}, {n2}, {n3}, {n4} filters and the rest having {n5}.",
"CNN code with {layer_count} layers of type {convolution_layer} and {pooling_function} pooling layers alternatively. The optimizer is {optimizer} and activation is {activation_function} and use {n1} filters for {k1} layer, {n2} filters for {k2} layer and {n3} filters for the rest..",
"To produce a code for classification in tensorflow. The number of layers is {layer_count} of type {convolution_layer} and {pooling_function} pooling. The activation is the standard {activation_function} with the usage of {optimizer}. The {k1}, {k2}, {k3} layers have {n1}, {n2}, {n3} filters and the rest have {n4}."
]

mapping = {0: 0, 1: 3, 2: 1, 3: 4, 4: 2, 5: 4, 6: 0, 7: 4, 8: 2, 9: 3}
words = ["", "first", "second", "third", "fourth", "fifth", "sixth", "seventh", "eight", "nineth", "tenth"]

convolution_layers = [
	"1D convolution",
	"2D convolution",
	"3D convolution",
	"1D separable convolution",
	"2D separable convolution",
	"3D separable convolution",
	"2D depthwise convolution",
	"2D transpose convolution",
	"3D transpose convolution"
]

pooling_functions = [
	"max pooling",
	"average pooling",
	"global max pooling",
	"global average pooling",
]

activation_functions = [
	"elu",
	"exponential",
	"gelu",
	"hard sigmoid",
	"linear",
	"relu",
	"selu",
	"sigmoid",
	"softplus",
	"softsign",
	"swish",
	"tanh"
]

optimizers = [
	"adadelta",
	"adagrad",
	"adam",
	"adammax",
	"ftrl",
	"nadam",
	"rms prop",
	"stochastic"
]

def format_string(template, count, layer_count, convolution_layer, pooling_function, activation_function, optimizer):
	out_string = template
	l = [i for i in range(1, layer_count+1)]
	output = ["" for i in range(0, layer_count+1)]
	output[0] = str(layer_count)
	if count == 0 or count == 6:
		n1 = random.randint(1, max_filter_count)
		out_string = out_string.format(layer_count=layer_count,n1 = n1, convolution_layer=convolution_layers[convolution_layer],
										 pooling_function=pooling_functions[pooling_function],
										 activation_function=activation_functions[activation_function],
										 optimizer=optimizers[optimizer])
		for i in range(1, layer_count + 1):
			output[i] = str(n1)
	elif count == 1 or count == 9:
		samples = random.sample(l, 3)
		n1 = random.randint(1, max_filter_count)
		n2 = random.randint(1, max_filter_count)
		n3 = random.randint(1, max_filter_count)
		n4 = random.randint(1, max_filter_count)
		out_string = out_string.format(layer_count=layer_count,
										k1=words[samples[0]],
										k2=words[samples[1]],
										k3=words[samples[2]],
										n1=n1,n2=n2,n3=n3,n4=n4,
										convolution_layer=convolution_layers[convolution_layer],
										 pooling_function=pooling_functions[pooling_function],
										 activation_function=activation_functions[activation_function],
										 optimizer=optimizers[optimizer])
		for i in range(1, layer_count + 1):
			output[i] = str(n4)	
		output[samples[0]] = str(n1)
		output[samples[1]] = str(n2)
		output[samples[2]] = str(n3)	

	elif count == 2:
		k1 = random.randint(1, layer_count)
		n1 = random.randint(1, max_filter_count)
		n2 = random.randint(1, max_filter_count)
		out_string = out_string.format(layer_count=layer_count,k1=k1,n1=n1,n2=n2, convolution_layer=convolution_layers[convolution_layer],
										 pooling_function=pooling_functions[pooling_function],
										 activation_function=activation_functions[activation_function],
										 optimizer=optimizers[optimizer])
		for i in range(1, layer_count + 1):
			output[i] = str(n2)	
		output[k1] = str(n1)

	elif count == 3 or count == 5 or count == 7:
		samples = random.sample(l, 4)
		n1 = random.randint(1, max_filter_count)
		n2 = random.randint(1, max_filter_count)
		n3 = random.randint(1, max_filter_count)
		n4 = random.randint(1, max_filter_count)
		n5 = random.randint(1, max_filter_count)

		out_string = out_string.format(layer_count=layer_count,
										k1=words[samples[0]],
										k2=words[samples[1]],
										k3=words[samples[2]],
										k4=words[samples[3]],
										n1=n1,n2=n2,n3=n3,n4=n4,n5=n5, convolution_layer=convolution_layers[convolution_layer],
										 pooling_function=pooling_functions[pooling_function],
										 activation_function=activation_functions[activation_function],
										 optimizer=optimizers[optimizer])

		for i in range(1, layer_count + 1):
			output[i] = str(n5)	
		output[samples[0]] = str(n1)
		output[samples[1]] = str(n2)
		output[samples[2]] = str(n3)
		output[samples[3]] = str(n4)

	elif count == 4 or count == 8:
		samples = random.sample(l, 2)
		n1 = random.randint(1, max_filter_count)
		n2 = random.randint(1, max_filter_count)
		n3 = random.randint(1, max_filter_count)
		out_string = out_string.format(layer_count=layer_count,
										k1=words[samples[0]],
										k2=words[samples[1]],
										n1=n1,n2=n2,n3=n3, convolution_layer=convolution_layers[convolution_layer],
										 pooling_function=pooling_functions[pooling_function],
										 activation_function=activation_functions[activation_function],
										 optimizer=optimizers[optimizer])
		for i in range(1, layer_count + 1):
			output[i] = str(n3)	
		output[samples[0]] = str(n1)
		output[samples[1]] = str(n2)

	return [out_string, output];


def generate_data():

	with open('data_all.csv', 'w') as csvfile:
		datawriter = csv.writer(csvfile, delimiter=',')
		datawriter.writerow(["input_string", "convolution_layer", "pooling_function", "activation_function", "optimizer", "r_convolution_layer", "r_pooling_function", "r_activation_function", "r_optimizer", "filters"])
		TRAIN_SAMPLES_COUNT = int(SAMPLES_COUNT);
		for i in range(0,TRAIN_SAMPLES_COUNT):

			template_num = random.choice(range(len(templates)))
			template = templates[template_num]
			count = mapping[template_num]
			layer_count = random.randint(count+1, max_layer_count)
			convolution_layer = random.choice(range(len(convolution_layers)))
			pooling_function = random.choice(range(len(pooling_functions)))
			activation_function = random.choice(range(len(activation_functions)))
			optimizer = random.choice(range(len(optimizers)))
			tmp = format_string(template, template_num, layer_count, convolution_layer, pooling_function, activation_function, optimizer)

			out_string = tmp[0]
			output = tmp[1]
			# print(out_string, output)

			## convolution_layer(9) || pooling_function(4) || activation_functions(12) || optimizers(8)
			## 0..8 || 9..12 || 13..24|| 25..32
			datawriter.writerow([out_string, convolution_layer, pooling_function, activation_function, optimizer, 
								convolution_layer, pooling_function + 9, activation_function + 9 + 4,
								optimizer + 9 + 4 + 12, ' '.join(output)])

def generate_token(state):
    return "STK" + str(state) + "STK"

def get_full_data(data_set):
	with open('data_all_filter.csv', 'w') as csvfile:
		datawriter = csv.writer(csvfile, delimiter=',')
		datawriter.writerow(["input_string", "classes"])
		data = data_set.to_numpy()
		data_new = []
		for row in data:
			curr_mapping = {}
			state = 0
			text = row[0]
			text = text.replace('.', ' ')
			text = text.replace(',', ' ')
			text = text.split(' ')
			new_text = []
			tmp = []
			# print(text)
			for i in text:
				if i.isnumeric():
					curr_mapping[i] = generate_token(state)
					state += 1
					new_text.append(curr_mapping[i])
				else:
					new_text.append(i)

			inp_string = ' '.join(new_text)

			out_text = row[9].split(' ')
			layer_count = int(out_text[0])
			for i in range(0, layer_count+1):
				tmp = []
				tmp.append(inp_string + " " + 'S' + str(i) + 'S')
				# print(out_text[i])
				# print(curr_mapping)
				# print(curr_mapping[out_text[i]])
				tmp.append(curr_mapping[out_text[i]])
				# print(tmp)
				datawriter.writerow(tmp)
			# exit()

def get_full_data_without_abs(data_set):
	with open('data_all_filter_no_abs.csv', 'w') as csvfile:
		datawriter = csv.writer(csvfile, delimiter=',')
		datawriter.writerow(["input_string", "classes"])
		data = data_set.to_numpy()
		data_new = []
		for row in data:

			inp_string = row[0]

			out_text = row[9].split(' ')
			layer_count = int(out_text[0])
			for i in range(0, layer_count+1):
				tmp = []
				tmp.append(inp_string + " " + 'S' + str(i) + 'S')
				tmp.append(out_text[i])
				# print(tmp)
				datawriter.writerow(tmp)
			# exit()

if __name__ == '__main__':
	# create_templates();
	# generate_data()
	data_set = pd.read_csv("data_all.csv")
	# data_set = get_full_data(data_set)
	data_set = get_full_data_without_abs(data_set)
