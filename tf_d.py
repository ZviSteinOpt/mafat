import numpy as np
import pandas as pd
from collections import Counter
import tensorflow as tf
from tensorflow import keras
from tensorflow.keras import layers
from tensorboard.plugins.hparams import api as hp
from imblearn.under_sampling import RandomUnderSampler
from scipy import io


data = pd.read_csv('/Users/zvistein/Downloads/mafat_wifi_challenge_training_set_v1.csv')

# orgenized the data 360X2
wind = 360
l = (len(data.RSSI_Right)-len(data.RSSI_Right)%wind)
b =  data.RSSI_Right[0:l]
a =  data.RSSI_Left[0:l]
a = np.append(a,b)
rss =  a.reshape(int(l/wind*2),wind)
rss_n = np.zeros((int(l/wind),wind,4))
rss_n[:,:,0] = (rss[0:int(l/wind),:]-np.min(rss[0:int(l/wind),:]))/(np.max(rss[0:int(l/wind),:])-np.min(rss[0:int(l/wind),:]))
rss_n[:,:,1] = (rss[int(l/wind):,:]-np.min(rss[int(l/wind):,:]))/(np.max(rss[int(l/wind):,:])-np.min(rss[int(l/wind):,:]))
# adding the substract betw the lobs
mx = np.max(rss[int(l/wind):,:]-rss[0:int(l/wind),:])
mn = np.min(rss[int(l/wind):,:]-rss[0:int(l/wind),:])
rss_n[:,:,2] = (rss[int(l/wind):,:]-rss[0:int(l/wind),:]-mn)/(mx-mn)
mx = np.max(rss[int(l/wind):,:]+rss[0:int(l/wind),:])
mn = np.min(rss[int(l/wind):,:]+rss[0:int(l/wind),:])
rss_n[:,:,3] = (rss[int(l/wind):,:]+rss[0:int(l/wind),:]-mn)/(mx-mn)

# the labels as well
b = data.Num_People[0:l]
num = b.values.reshape(int(l/wind),wind)
gt = np.zeros(int(l/wind))
for i in np.arange(0,int(l/wind)):
 n = num[i,:]
 b = Counter(n)
 gt[i]  = b.most_common(1)[0][0]


# undersumpeling treatment
rus = RandomUnderSampler(sampling_strategy={0: 2110, 1: 988, 2: 3000, 3: 1096})
d_s = np.zeros((len(rss_n),2))
d_s[:,0] = np.arange(0,len(rss_n))
x_u,y_u = rus.fit_resample(d_s,gt)
x_u = x_u.astype(int)
gt   = gt[x_u[:,0]]
data_ds = rss_n[x_u[:,0],:]


# data loader
train_size = int(0.8 * len(data_ds))
full_dataset = tf.data.Dataset.from_tensor_slices((data_ds,gt)).shuffle(4000)
#full_dataset = full_dataset.shuffle()

train_ds = full_dataset.take(train_size).batch(5500)
valid_ds = full_dataset.skip(train_size).batch(1200)

model = tf.keras.models.Sequential([
  tf.keras.layers.Flatten(input_shape=(360, 4)),
  tf.keras.layers.Dense(128, activation='relu'),
  tf.keras.layers.Dropout(0.2),
  tf.keras.layers.Dense(10)
])


adm=tf.keras.optimizers.Adam(lr=0.1, beta_1=0.9, beta_2=0.999, epsilon=1e-08)
model.compile(loss='categorical_crossentropy', metrics=['accuracy'],optimizer=adm)

for images, labels in train_ds:
   for valid_images, valid_labels in valid_ds:
      model.fit(images,labels,batch_size=25, epochs=10, )

