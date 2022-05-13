
convolution_layers = [
	"Conv1D",
	"Conv2D",
	"Conv3D",
	"SeparableConv1D",
	"SeparableConv2D",
	"SeparableConv3D",
	"DepthwiseConv2D",
	"Conv2DTranspose",
	"Conv3DTranspose"
]

pooling_functions = [
	"MaxPooling",
	"AveragePooling",
	"GlobalMaxPooling",
	"GlobalAveragePooling",
]

activation_functions = [
	"elu",
	"exponential",
	"gelu",
	"hard_sigmoid",
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
	"rmsprop",
	"sgd"
]

def cnn_model(layer_count, conv_fn, pooling_fn, activation_fn, optimizer, filters):
	cnn_model_st = "import tensorflow as tf\n"
	cnn_model_st += "from tensorflow.keras import layers, models\n\n"

	cnn_model_st += "model = models.Sequential()\n"

	for i in range(0, layer_count):
		cnn_model_st += "model.add(layers." + convolution_layers[conv_fn] + "(" + filters[i]  + ", kernel_size, activation='" + activation_functions[activation_fn] + "'))\n"
		if "1" in convolution_layers[conv_fn]:
			layer_type = "1D"
		elif "2" in convolution_layers[conv_fn]:
			layer_type = "2D"
		elif "3" in convolution_layers[conv_fn]:
			layer_type = "3D"
		
		cnn_model_st += "model.add(layers." + pooling_functions[pooling_fn] + layer_type + "())\n"
	
	cnn_model_st += "model.add(layers.Flatten())\n"
	cnn_model_st += "model.add(layers.Dense(output_class_count))\n"

	cnn_model_st += "model.compile(optimizer='" + optimizers[optimizer] + "', loss=tf.keras.losses.SparseCategoricalCrossentropy(), metrics=['accuracy'])\n"


	print(cnn_model_st)
	