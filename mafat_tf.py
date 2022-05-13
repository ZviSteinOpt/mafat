import numpy as np
import pandas as pd
from collections import Counter

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
