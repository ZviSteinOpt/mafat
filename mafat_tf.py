import numpy as np
import pandas as pd
from collections import Counter
import tensorflow as tf
from tensorflow import keras
from tensorflow.keras import layers

data = pd.read_csv('/Users/zvistein/Downloads/mafat_wifi_challenge_training_set_v1.csv')
train = data.sample(frac=0.8,random_state=0)
test  = data.drop(train.index)
l = (len(data.RSSI_Right)-len(data.RSSI_Right)%360)
b =  data.RSSI_Right[0:l]
a =  data.RSSI_Left[0:l]
a = np.append(a,b)
rss =  a.reshape(int(l/360*2),360)
rss_n = np.zeros((int(l/360),360,2))
rss_n[:,:,0] = rss[0:int(l/360),:]
rss_n[:,:,1] = rss[int(l/360):,:]

b = data.Num_People[0:l]
num = b.values.reshape(int(l/360),360)
gt = np.zeros(int(l/360))
for i in np.arange(0,int(l/360)):
 n = num[i,:]
 b = Counter(n)
 gt[i] = b.most_common(1)[0][0]



train_ds = tf.data.Dataset.from_tensor_slices((rss_n,gt))

class cnnMA(keras.Model):
    def __init__(self):
        super(cnnMA, self).__init__()
        self.conv1 = layers.Conv2D(48, (2,1), activation='relu', strides=(1, 1))
        # Shape= (b_s,12,50,50)
        self.bn1 = layers.BatchNormalization( axis = 1)
        # Shape= (b_s,12,50,50)

        # Input shape= (b_s,1,50,50)
        self.flatten = layers.Flatten()

        self.fc1 = layers.Dense(36000, activation='relu')
        self.bn2 = layers.BatchNormalization()
        self.fc2 = layers.Dense(1000, activation='relu')
        self.bn3 = layers.BatchNormalization()
        self.fc3 = layers.Dense(100, activation='relu')
        self.bn4 = layers.BatchNormalization()
        self.fc4 = layers.Dense(2, activation='relu')


        # Feed forwad function

    def call(self, input):
        output = self.conv1(input)
        output = self.bn1(output)

        output = self.flatten(output)

        output = self.fc1(output)
        output = self.bn2(output)
        output = self.fc2(output)
        output = self.bn3(output)
        output = self.fc3(output)
        output = self.bn4(output)
        output = self.fc4(output)

        return output

model = cnnMA()

loss_fun  = keras.losses.SparseCategoricalCrossentropy(from_logits = True)
optimizer = tf.keras.optimizers.Adam()
optimizer.lr = 0.001

train_loss = tf.keras.metrics.Mean(name='train_loss')
train_accuracy = tf.keras.metrics.SparseCategoricalAccuracy(name='train_accuracy')

test_loss = tf.keras.metrics.Mean(name='test_loss')
test_accuracy = tf.keras.metrics.SparseCategoricalAccuracy(name='test_accuracy')

def train_step(seq, labels):
  with tf.GradientTape() as tape:
    # training=True is only needed if there are layers with different
    # behavior during training versus inference (e.g. Dropout).
    seq1 = tf.expand_dims(seq, axis=0)
    seq2 = tf.expand_dims(seq1, axis=-1)

    predictions = model(seq2, training=True)
    loss = loss_fun(labels, predictions)
  gradients = tape.gradient(loss, model.trainable_variables)
  optimizer.apply_gradients(zip(gradients, model.trainable_variables))

  train_loss(loss)
  train_accuracy(labels, predictions)

EPOCHS = 5

for epoch in range(EPOCHS):
      # Reset the metrics at the start of the next epoch
      train_loss.reset_states()
      train_accuracy.reset_states()
      test_loss.reset_states()
      test_accuracy.reset_states()

      for images, labels in train_ds:
          train_step(images, labels)

      for test_images, test_labels in test_ds:
          test_step(test_images, test_labels)

      print(
          f'Epoch {epoch + 1}, '
          f'Loss: {train_loss.result()}, '
          f'Accuracy: {train_accuracy.result() * 100}, '
          f'Test Loss: {test_loss.result()}, '
          f'Test Accuracy: {test_accuracy.result() * 100}'
      )




