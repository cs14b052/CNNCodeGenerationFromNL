import numpy as np
import tensorflow as tf
import pandas as pd
from tensorflow.keras import layers
import os
import random

os.environ['TF_CPP_MIN_LOG_LEVEL'] = '2'

import matplotlib.pyplot as plt

def plot_graphs(history, metric):
  plt.plot(history.history[metric])
  plt.plot(history.history['val_'+metric], '')
  plt.xlabel("Epochs")
  plt.ylabel(metric)
  plt.legend([metric, 'val_'+metric])


OPTIONS_TO_CHOOSE = 2

def get_full_data(data_set):
    options = ['SConvTypeS', 'SOptimizerS', 'SPoolingFnS', 'SActivationFnS']

    data = data_set.to_numpy()
    data_new = []
    for row in data:
        samples = random.sample([x for x in range(0, len(options))], OPTIONS_TO_CHOOSE)
        for i in samples:
            tmp_list = []
            tmp_list.append(row[0] + " " + options[i])
            if i == 0:
                tmp_list.append(row[5])
            elif i == 1:
                tmp_list.append(row[8])
            elif i == 2:
                tmp_list.append(row[6])
            elif i == 3:
                tmp_list.append(row[7])
            data_new.append(tmp_list)
    df = pd.DataFrame(data_new, columns=['input_string', 'classes'])
    return df


TRAIN_DATA_PER = 0.8
SEED = 200
VOCAB_SIZE = 1000
LEARNING_RATE = 1e-4
metrics = ['accuracy']
EPOCHS = 15
BUFFER_SIZE = 10000
BATCH_SIZE = 256
STEPS_PER_EPOCH = 1000

# train_data = pd.read_csv("train_data.csv")
# train_data = train_data[['input_string', 'optimizer']]
# test_data = pd.read_csv("test_data.csv")
# test_data = test_data[['input_string', 'optimizer']]

data_set = pd.read_csv("data_all.csv")
# data_set = data_set[['input_string', 'optimizer']]

data_set = get_full_data(data_set)

train_data = data_set.sample(frac=TRAIN_DATA_PER,random_state=SEED)
test_data = data_set.drop(train_data.index)

# train_data = tf.data.Dataset.from_tensor_slices(dict(train_data))
# test_data = tf.data.Dataset.from_tensor_slices(dict(test_data))

# train_data = train_data.shuffle(BUFFER_SIZE).batch(BATCH_SIZE)
# test_data = test_data.batch(BATCH_SIZE)


encoder = tf.keras.layers.TextVectorization(
    max_tokens=VOCAB_SIZE)

encoder.adapt(train_data['input_string'].to_numpy())

model = tf.keras.Sequential([
	tf.keras.Input(shape=(1,), dtype=tf.string, name='input_string'),
    encoder,
    tf.keras.layers.Embedding(
        input_dim=len(encoder.get_vocabulary()),
        output_dim=66,
        # Use masking to handle the variable sequence lengths
        mask_zero=True,
        name='embedding_layer'),
    tf.keras.layers.Bidirectional(tf.keras.layers.LSTM(66), name='bidirectional_LSTM_layer'),
    tf.keras.layers.Dense(66, activation='relu', name='dense_layer'),
    tf.keras.layers.Dense(33, activation='softmax', name='classes')
])

# sample_text = ("Generate CNN code for classification with layer count of {layer_count} and use {convolution_layer} layers. Use {activation_function} as activation function and {optimizer} optimizer. The pooling layer is {pooling_function}"
# 	.format(layer_count=4, convolution_layer='1D convolution', activation_function='relu', optimizer='adam', pooling_function='max pooling'))
# predictions = model.predict(np.array([sample_text]))
# print(predictions[0])

model.compile(loss=tf.keras.losses.SparseCategoricalCrossentropy(from_logits=False),
              optimizer=tf.keras.optimizers.Adam(LEARNING_RATE),
              metrics=metrics)

print ('model compiled')

history = model.fit(x=train_data['input_string'].to_numpy(),
					y=train_data['classes'].to_numpy(),
					epochs=EPOCHS,
					steps_per_epoch=STEPS_PER_EPOCH,
                    # batch_size=BATCH_SIZE,
                    validation_data=(test_data['input_string'].to_numpy(), test_data['classes'].to_numpy()),
                    # validation_steps=30
                    )

print ('model has been fit')
model.save('rnn_task1')
model.summary()

test_loss, test_acc = model.evaluate(x=test_data['input_string'].to_numpy(), y=test_data['classes'].to_numpy())

print('Test Loss:', test_loss)
print('Test Accuracy:', test_acc)

plt.figure(figsize=(16, 8))
plt.subplot(1, 2, 1)
plot_graphs(history, 'accuracy')
plt.ylim(None, 1)
plt.subplot(1, 2, 2)
plot_graphs(history, 'loss')
plt.ylim(0, None)
plt.savefig('rnn_task1.png')


