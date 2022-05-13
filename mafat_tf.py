import numpy as np
import pandas as pd
from collections import Counter
import tensorflow as tf
from tensorflow import keras
from tensorflow.keras import layers

data = pd.read_csv('/Users/zvistein/Downloads/mafat_wifi_challenge_training_set_v1.csv')
l = (len(data.RSSI_Right)-len(data.RSSI_Right)%360)
b =  data.RSSI_Right[0:l]
a =  data.RSSI_Left[0:l]
a = np.append(a, b)
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

train_data = []
for i in range(len(rss_n)):
   train_data.append([rss_n[i], gt[i]])

model = keras.models.Sequential()

class cnnMA(keras.Model):
    def __init__(self):
        super(cnnMA, self).__init__()
        self.conv1 = layers.Conv2D(48, 3, activation='relu') # Conv2d(in_channels=1, out_channels=48, kernel_size=[1,2], stride=1, padding=0)
        # Shape= (b_s,12,50,50)
        self.bn1 = layers.BatchNormalization(num_features=48)
        # Shape= (b_s,12,50,50)

        # Input shape= (b_s,1,50,50)
        self.flatten = layers.Flatten()

        self.fc1 = layers.Dense(in_features=48*360*1, out_features=36000)
        self.bn2 = layers.BatchNormalization(36000)
        self.fc2 = layers.Linear(in_features=36000, out_features = 1000)
        self.bn3 = layers.BatchNormalization(1000)
        self.fc3 = layers.Dense(in_features=1000, out_features = 100)
        self.bn4 = layers.BatchNormalization(100)
        self.fc4 = layers.Dense(in_features=100, out_features = 2)


        # Feed forwad function

    def call(self, input):
        output = self.conv1(input)
        output = self.bn1(output)
        output = self.Lrelu(output)

        output = self.Flatten(output)

        output = self.fc1(output)
        output = self.bn2(output)
        output = self.Lrelu(output)
        output = self.fc2(output)
        output = self.bn3(output)
        output = self.Lrelu(output)
        output = self.fc3(output)
        output = self.bn4(output)
        output = self.Lrelu(output)
        output = self.fc4(output)

        return output
model = cnnMA()
loss  = keras.losses.SparseCategoricalCrossentropy(from_logits = True)
optimizer = tf.keras.optimizers.Adam()
optimizer.lr = 0.001


def train_step(images, labels):
  with tf.GradientTape() as tape:
    # training=True is only needed if there are layers with different
    # behavior during training versus inference (e.g. Dropout).
    predictions = model(images, training=True)
    loss = loss_object(labels, predictions)
  gradients = tape.gradient(loss, model.trainable_variables)
  optimizer.apply_gradients(zip(gradients, model.trainable_variables))

  train_loss(loss)
  train_accuracy(labels, predictions)