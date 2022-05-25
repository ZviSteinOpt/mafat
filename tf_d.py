import tensorflow as tf
from tensorflow import keras
from tensorflow.keras import layers
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
from collections import Counter
import math

data = pd.read_csv('/Users/zvistein/Downloads/mafat_wifi_challenge_training_set_v1.csv')
left  = data.RSSI_Left
left = left.to_numpy()
num_people = data.Num_People
num_people = num_people.to_numpy()
time = data.Time
room_num = data.Room_Num
id = data.Device_ID
right = data.RSSI_Right
right = right.to_numpy()
dif   = right - left


plt.plot(time[0:7632*360], dif[0:7632*360], 'bo')
plt.grid()
plt.show()


from numpy.ma.core import count
un_Ids     = np.unique(id) # Dividing by Device
fin_size   = int(len(dif)/360) # Usable Size
fin_dif    = dif[0:fin_size*360]
fin_left   = left[0:fin_size*360]
fin_right  = right[0:fin_size*360]
fin_num_people = num_people[0:fin_size*360]
fin_num_people[fin_num_people!=0] = 1


fin_dif = fin_dif.reshape(7632,360)
fin_dif = fin_dif.astype('float64')
fin_num_people = fin_num_people.reshape(7632,360)
fin_left = fin_left.reshape(7632,360)
fin_right = fin_right.reshape(7632,360)
fin_labels = np.zeros(fin_size)

indy = np.array([])
for i in range(0,fin_size):
    temp = fin_num_people[i,:]
    temp = Counter(temp)
    num = temp.most_common(1)[0][0]
    if num!=0:
        fin_labels[i] = 1
    else:
        fin_labels[i] = 0
    if np.std(fin_dif[i,:])!=0:
        indy = np.append(indy,i)

# Cleaning Trash Data std=0
fin_left = fin_left[indy.astype('int'),:]
fin_right = fin_right[indy.astype('int'),:]
fin_labels = fin_labels[indy.astype('int')]
fin_dif = fin_dif[indy.astype('int')]
fin_dif = fin_dif.astype('float64')

# Normalizing data. Subtracting mean, dividing std
for i in range(0,len(indy)):
  fin_dif[i,:]-=np.mean(fin_dif[i,:])
  fin_dif[i,:]/=np.std(fin_dif[i,:])

model = tf.keras.models.Sequential([
    tf.keras.layers.Conv1D(10,3,activation='relu',input_shape = (360,1)),
    tf.keras.layers.BatchNormalization(),
    tf.keras.layers.Flatten(),
    tf.keras.layers.Dense(512,activation='relu'),
    tf.keras.layers.BatchNormalization(),
    tf.keras.layers.Dense(64,activation='relu'),
    tf.keras.layers.BatchNormalization(),
    tf.keras.layers.Dense(1,activation='sigmoid')]
)
model.summary()
model.compile(optimizer='rmsprop',loss='binary_crossentropy',metrics = ['accuracy'])
model.fit(fin_dif,fin_labels,epochs = 10,verbose = 1,validation_split=0.1)