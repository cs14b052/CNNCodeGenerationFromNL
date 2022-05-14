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



TRAIN_DATA_PER = 0.8
SEED = 200
VOCAB_SIZE = 1000
LEARNING_RATE = 1e-4
metrics = ['accuracy']
EPOCHS = 20
BUFFER_SIZE = 10000
BATCH_SIZE = 256
STEPS_PER_EPOCH = 1000
DATA_SIZE = 50000

def generate_token(state):
    return "STK" + str(state) + "STK"

def get_full_data(data_set):
    data = data_set.to_numpy()
    data_new = []
    for row in data:
        mapping = {}
        state = 0
        text = row[0]
        text = text.replace('.', ' ')
        text = text.replace(',', ' ')
        text = text.split(' ')
        new_text = []
        tmp = []
        for i in text:
            if i.isnumeric():
                mapping[i] = generate_token(state)
                state += 1
                new_text.append(mapping[i])
            else:
                new_text.append(i)

        inp_string = ' '.join(new_text)
        tmp.append(inp_string + " " + 'SLayerS')

        out_text = row[1].split(' ')
        layer_count = int(out_text[0])
        tmp.append(out_text)
        data_new.append(tmp)
        for i in range(1, layer_count+1):
            tmp = []
            tmp.append(inp_string + " " + 'S' + str(i) + 'S')
            # print(text, out_text)
            tmp.append(mapping[out_text[i]])
            data_new.append(tmp)

    df = pd.DataFrame(data_new, columns=['input_string', 'classes'])
    return df

class MyModel(tf.keras.Model):
    """docstring for MyModel"""
    def __init__(self, embedding_dim, rnn_units):
        super().__init__(self)
        self.embedding_dim = tf.keras.layers.Embedding(VOCAB_SIZE, embedding_dim)
        self.gru = tf.keras.layers.GRU(rnn_units, return_sequences=True, return_state=True)
        self.dense = tf.keras.layers.Dense(VOCAB_SIZE)

    def call(self, inputs, states=None, return_state=False, training=False):
        x = inputs
        x = self.embedding(x, training=training)
        if states is None:
          states = self.gru.get_initial_state(x)
        x, states = self.gru(x, initial_state=states, training=training)
        x = self.dense(x, training=training)

        if return_state:
          return x, states
        else:
          return x

def convt_to_dict(l_vocab):
    mapping = {}
    for i in range(0, len(l_vocab)):
        mapping[l_vocab[i]] = i

    return mapping

def encode_y(vocab_dict, y_true):
    y_changed = []
    for i in y_true:
        y_changed.append(vocab_dict[str(i).lower()])
    return y_changed



data_set = pd.read_csv("data_all_filter_no_abs.csv")
# data_set = get_full_data(data_set)
print('data generated')
data_set = data_set.head(DATA_SIZE)
train_data = data_set.sample(frac=TRAIN_DATA_PER,random_state=SEED)
test_data = data_set.drop(train_data.index)

encoder = tf.keras.layers.TextVectorization(
    max_tokens=VOCAB_SIZE)

encoder.adapt(train_data['input_string'].to_numpy())

VOCAB_SIZE = len(encoder.get_vocabulary())

embedding_dim = 256
rnn_units = 16
# model = MyModel(
#     embedding_dim=embedding_dim,
#     rnn_units=rnn_units)
print('data encoded')
model = tf.keras.Sequential([
    tf.keras.Input(shape=(1,), dtype=tf.string, name='input_string'),
    encoder,
    tf.keras.layers.Embedding(
        input_dim=len(encoder.get_vocabulary()),
        output_dim=1024,
        # Use masking to handle the variable sequence lengths
        mask_zero=True,
        name='embedding_layer'),
    tf.keras.layers.Bidirectional(tf.keras.layers.LSTM(1024), name='bidirectional_LSTM_layer'),
    tf.keras.layers.Dense(1024, activation='relu', name='dense_layer'),
    tf.keras.layers.Dense(VOCAB_SIZE, activation='softmax', name='classes')
])

y_true = train_data['classes'].to_numpy()
y_test = test_data['classes'].to_numpy()
vocab_dict = convt_to_dict(encoder.get_vocabulary())
y_true = encode_y(vocab_dict, y_true)
y_test = encode_y(vocab_dict, y_test)

model.compile(loss=tf.keras.losses.SparseCategoricalCrossentropy(from_logits=False),
              optimizer=tf.keras.optimizers.Adam(LEARNING_RATE),
              metrics=metrics)

print ('model compiled')

# Directory where the checkpoints will be saved
checkpoint_dir = './training_checkpoints_no_abs'
# Name of the checkpoint files
checkpoint_prefix = os.path.join(checkpoint_dir, "ckpt_{epoch}")

checkpoint_callback = tf.keras.callbacks.ModelCheckpoint(
    filepath=checkpoint_prefix)


history = model.fit(x=train_data['input_string'].to_numpy(),
					y=np.array(y_true),
					epochs=EPOCHS,
					steps_per_epoch=STEPS_PER_EPOCH,
                    # batch_size=BATCH_SIZE,
                    validation_data=(test_data['input_string'].to_numpy(), np.array(y_test)),
                    # validation_steps=30
                    callbacks=[checkpoint_callback]
                    )

print ('model has been fit')
model.save('baseline')
model.summary()

test_loss, test_acc = model.evaluate(x=test_data['input_string'].to_numpy(), y=np.array(y_test))

print('Test Loss:', test_loss)
print('Test Accuracy:', test_acc)

plt.figure(figsize=(16, 8))
plt.subplot(1, 2, 1)
plot_graphs(history, 'accuracy')
plt.ylim(None, 1)
plt.subplot(1, 2, 2)
plot_graphs(history, 'loss')
plt.ylim(0, None)
plt.savefig('baseline.png')


