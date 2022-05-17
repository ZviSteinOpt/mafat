import numpy as np
import pandas as pd
from collections import Counter
import tensorflow as tf
from tensorflow import keras
from tensorflow.keras import layers
from tensorboard.plugins.hparams import api as hp
from imblearn.under_sampling import RandomUnderSampler
from scipy import io

mat = io.loadmat('/Users/zvistein/Desktop/matlab.mat')
fich = mat['fich']
lab = mat['lab']
lab = np.squeeze(lab)

rus = RandomUnderSampler(sampling_strategy=1)
data = pd.read_csv('/Users/zvistein/Downloads/mafat_wifi_challenge_training_set_v1.csv')
wind = 360
l = (len(data.RSSI_Right)-len(data.RSSI_Right)%wind)
b =  data.RSSI_Right[0:l]
a =  data.RSSI_Left[0:l]
a = np.append(a,b)
rss =  a.reshape(int(l/wind*2),wind)
rss_n = np.zeros((int(l/wind),wind,2))
rss_n[:,:,0] = rss[0:int(l/wind),:]
rss_n[:,:,1] = rss[int(l/wind):,:]

b = data.Num_People[0:l]
num = b.values.reshape(int(l/wind),wind)
gt = np.zeros(int(l/wind))
for i in np.arange(0,int(l/wind)):
 n = num[i,:]
 b = Counter(n)
 gt[i]  = np.sign(b.most_common(1)[0][0])

d_s = np.zeros((len(rss_n),2))
d_s[:,0] = np.arange(0,len(rss_n))
x_u,y_u = rus.fit_resample(d_s,gt)
x_u = x_u.astype(int)
gt   = gt[x_u[:,0]]
data_ds = rss_n[x_u[:,0],:]
train_size = int(0.8 * len(data_ds))
full_dataset = tf.data.Dataset.from_tensor_slices((data_ds,gt)).shuffle(4000)
#full_dataset = full_dataset.shuffle()

train_ds = full_dataset.take(train_size).batch(15)
valid_ds = full_dataset.skip(train_size).batch(15)
class cnnMA(keras.Model):
    def __init__(self):
        super(cnnMA, self).__init__()
        self.conv1 = layers.Conv2D(3, (10,2), activation='relu', strides=(1, 1))
        # Shape= (b_s,12,50,50)
        self.bn1 = layers.BatchNormalization( axis = 2)
        # Shape= (b_s,12,50,50)

        # Input shape= (b_s,1,50,50)
        self.flatten = layers.Flatten()
        self.fc1 = layers.Dense(90000, activation='LeakyReLU')
        self.bn2 = layers.BatchNormalization()
        self.fc2 = layers.Dense(10000, activation='LeakyReLU')
        self.bn3 = layers.BatchNormalization()
        self.fc3 = layers.Dense(1000, activation='LeakyReLU')
        self.bn4 = layers.BatchNormalization()
        self.fc4 = layers.Dense(100, activation='LeakyReLU')
        self.bn5 = layers.BatchNormalization()
        self.fc5 = layers.Dense(250, activation='LeakyReLU')
        self.bn6 = layers.BatchNormalization()
        self.fc6 = layers.Dense(2, activation='LeakyReLU')
        self.s   = layers.Softmax(axis=-1)

        # Feed forwad function

    def call(self, input):
        #output = self.conv1(input)
        #output = self.bn1(output)


        output = self.flatten(input)
        output = self.fc1(output)
        output = self.bn2(output)
        output = self.fc2(output)
        output = self.bn3(output)
        output = self.fc3(output)
        output = self.bn4(output)
        output = self.fc4(output)
        output = self.bn5(output)
        output = self.fc5(output)
        output = self.bn6(output)
        output = self.fc6(output)
        #output = self.s(output)
        return output

model = cnnMA()

loss_fun  = keras.losses.SparseCategoricalCrossentropy(from_logits = True)
optimizer = tf.keras.optimizers.SGD()
optimizer.lr = 0.1
train_loss = tf.keras.metrics.Mean(name='train_loss')
train_accuracy = tf.keras.metrics.SparseCategoricalAccuracy(name='train_accuracy')

valid_loss = tf.keras.metrics.Mean(name='test_loss')
valid_accuracy = tf.keras.metrics.SparseCategoricalAccuracy(name='test_accuracy')

def train_step(seq, labels):
  with tf.GradientTape() as tape:
    # training=True is only needed if there are layers with different
    # behavior during training versus inference (e.g. Dropout).
    seq1 = tf.expand_dims(seq, axis=1)
    seq2 = tf.expand_dims(seq1, axis=-1)

    predictions = model(seq2, training=True)
    loss = loss_fun(labels, predictions)
    # print(loss)
  gradients = tape.gradient(loss, model.trainable_variables)
  optimizer.apply_gradients(zip(gradients, model.trainable_variables))

  train_loss(loss)
  train_accuracy(labels, predictions)

def valid_step(seq, labels):
    # training=False is only needed if there are layers with different
    # behavior during training versus inference (e.g. Dropout).
    seq1 = tf.expand_dims(seq, axis=1)
    seq2 = tf.expand_dims(seq1, axis=-1)
    predictions = model(seq2, training=False)
    t_loss = loss_fun(labels, predictions)

    valid_loss(t_loss)
    valid_accuracy(labels, predictions)


EPOCHS = 100

for epoch in range(EPOCHS):
      # Reset the metrics at the start of the next epoch
      train_loss.reset_states()
      train_accuracy.reset_states()
      valid_loss.reset_states()
      valid_accuracy.reset_states()

      for images, labels in train_ds:
          train_step(images, labels)

      for valid_images, valid_labels in valid_ds:
          valid_step(valid_images, valid_labels)

      print(
          f'Epoch {epoch + 1}, '
          f'Loss: {train_loss.result()}, '
          f'Accuracy: {train_accuracy.result() * 100}, '
          f'Test Loss: {valid_loss.result()}, '
          f'Test Accuracy: {valid_accuracy.result() * 100}'
      )
