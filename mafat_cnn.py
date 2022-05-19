# just a comment
import numpy as np
import pandas as pd
from collections import Counter
import matplotlib.pyplot as plt
import torch
import torch.nn as nn
from torch.utils.data import DataLoader,WeightedRandomSampler,random_split
import time

if torch.backends.mps.is_available():
    device = torch.device('mps')
else:
    device = torch.device('cpu')

def get_loader(DataSets,class_weights):

    # oversampling the unbalanced data
    #class_weights = [3,3,1,3] # the balance ratio is 1 to 5
    sample_weights = [0]*len(DataSets)

    for idx, (data,label) in enumerate(DataSets):
        class_weight = class_weights[int(label)]
        sample_weights[idx] = class_weight

    sampler = WeightedRandomSampler(sample_weights, num_samples = len(sample_weights), replacement=True)

    # Dataloader
    train_loader = DataLoader(DataSets,
        batch_size=100,sampler = sampler)

    return train_loader



data = pd.read_csv('/Users/zvistein/Downloads/mafat_wifi_challenge_training_set_v1.csv')
wind = 360
rm = 3
Dr = data.RSSI_Right[data.Room_Num == rm]
Dl = data.RSSI_Left[data.Room_Num == rm]
La = data.Num_People[data.Room_Num == rm]
l = (len(Dr)-len(Dr)%wind)
b =  Dr[0:l]
a =  Dl[0:l]
a = np.append(a,b)
rss =  a.reshape(int(l/wind*2),wind)
rss_n = np.zeros((int(l/wind),wind,1))
stdz = []
for i in range(len(rss_n)):
    mn = np.mean(rss[i,:])
    mx0 = np.std(rss[i,:])
    rss_n[i,:,0] = (rss[i,:]-mn)/mx0
    mn = np.mean(rss[i+int(l/wind),:])
    mx1 = np.std(rss[i+int(l/wind),:])
    rss_n[i,:,0] = (rss[i+int(l/wind),:]-mn)/mx1

    # adding the substract betw the lobs
    temp = rss[i,:]+rss[i+int(l/wind),:]
    mn = np.mean(temp)
    mx2 = np.std(temp)
    rss_n[i,:,0] = (temp-mn)/mx2
    temp = rss[i,:]-rss[i+int(l/wind),:]
    mn = np.mean(temp)
    mx3 = np.std(temp)
    rss_n[i,:,0] = (temp-mn)/mx3
    if mx3==0 or mx1==0:
        stdz.append(i)
    else:
        gg = 1

rss_n = np.delete(rss_n,stdz,0)

b = La[0:l]
num = b.values.reshape(int(l/360),360)
gt = np.zeros(int(l/360))
for i in np.arange(0,int(l/360)):
 n = num[i,:]
 b = Counter(n)
 gt[i] = b.most_common(1)[0][0]

weit = [0.25/(sum(gt==0)/len(gt)) ,0.25/(sum(gt==1)/len(gt)) ,0.25/(sum(gt==2)/len(gt)) ,0.25/(sum(gt==3)/len(gt))]
weit = np.round(weit/min(weit))
train_data = []
train_data_of = []
flag = np.zeros(4)
count = 0
for i in range(len(rss_n)):
   train_data.append([rss_n[i], gt[i]])
   flag[int(gt[i])] = flag[int(gt[i])]+1
   if flag[int(gt[i])]<100:
       train_data_of.append([rss_n[i], gt[i]])
       count = count+1


class FN(nn.Module):
    def __init__(self, num_classes = 4):
        super(FN, self).__init__()

        # Output size after convolution filter
        # ((w-f+2P)/s) +1
        self.conv1 = nn.Conv2d(in_channels=1, out_channels=24, kernel_size=[1,2], stride=1, padding=0)
        # Shape= (b_s,12,50,50)
        self.bn1 = nn.BatchNorm2d(num_features=24)
        # Shape= (b_s,12,50,50)

        # Input shape= (b_s,1,50,50)

        self.fc1 = nn.Linear(in_features=360*1, out_features=36000)
        self.bn2 = nn.BatchNorm1d(36000)
        self.fc2 = nn.Linear(in_features=36000, out_features = 1000)
        self.bn3 = nn.BatchNorm1d(1000)
        self.fc3 = nn.Linear(in_features=1000, out_features = 1000)
        self.bn4 = nn.BatchNorm1d(1000)
        self.fc4 = nn.Linear(in_features=1000, out_features = 500)
        self.bn5 = nn.BatchNorm1d(500)
        self.fc5 = nn.Linear(in_features=500, out_features = 250)
        self.bn6 = nn.BatchNorm1d(250)
        self.fc6 = nn.Linear(in_features=250, out_features = 100)
        self.bn7 = nn.BatchNorm1d(100)
        self.fc7 = nn.Linear(in_features=100, out_features = 4)

        self.relu = nn.ReLU()
        self.Lrelu = nn.LeakyReLU()

        # Feed forwad function

    def forward(self, input):
        # output = self.conv1(input)
        # output = self.bn1(output)
        # output = self.Lrelu(output)

        output = input.view(-1, 1*360)
        # output  = output.view(-1, 12*360 * 3)
        # output = torch.cat(( output.permute(1,0) , input.permute(1,0) ),0)
        # output = output.permute(1, 0)
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
        output = self.bn5(output)
        output = self.relu(output)
        output = self.fc5(output)
        output = self.bn6(output)
        output = self.Lrelu(output)
        output = self.fc6(output)
        output = self.bn7(output)
        output = self.Lrelu(output)
        output = self.fc7(output)



        return output


test_count  = len(rss_n)//9
train_count = len(rss_n)-test_count
# train_count = np.floor(count*0.8)
# test_count  = count-np.floor(count*0.8)

train_sets, test_setes = random_split(train_data,[train_count,test_count])
# train_sets, test_setes = random_split(train_data_of,[int(train_count),int(test_count)])
test_loader = DataLoader(test_setes,
    batch_size=500, shuffle=True)
train_loader = get_loader(train_sets,weit)
# train_loader = DataLoader(train_sets,
#     batch_size=250, shuffle=True)

torch.manual_seed(3407)
model = FN(num_classes=1).to(device)

optimizer     = torch.optim.Adam(model.parameters(), lr=0.01)
scheduler = torch.optim.lr_scheduler.StepLR(optimizer, step_size=150, gamma=(0.9))

loss_function = nn.CrossEntropyLoss()

num_epochs = 1000
s_loss = torch.zeros(num_epochs*len(train_loader))
idx = 0

for epoch in range(num_epochs):

    # Evaluation and training on training dataset
    model.train()
    train_accuracy = 0.0
    train_loss = 0.0

    for i, (seq, labels) in enumerate(train_loader):
        optimizer.zero_grad()
        seq = seq[None, :]
        seq = seq.permute(1, 0, 2, 3)
        seq = seq.float()
        seq = seq.to(device)
        outputs = model(seq)
        labels = labels.long()
        labels = labels.to(device)
        loss = loss_function(outputs, labels)
        s_loss[idx] = loss
        idx = idx+1
        loss.backward()
        optimizer.step()
        scheduler.step()
        train_loss += loss.cpu().data * seq.size(0)
        _, prediction = torch.max(outputs.data, 1)

        train_accuracy += int(torch.sum(prediction == labels.data))

    train_accuracy = train_accuracy / train_count
    train_loss = train_loss / train_count
    # Evaluation on testing dataset
    model.eval()

    test_accuracy = 0.0
    for i, (seq, labels) in enumerate(test_loader):
        seq = seq[None, :]
        seq = seq.permute(1, 0, 2, 3)
        seq = seq.float()
        seq = seq.to(device)
        labels = labels.long()
        labels = labels.to(device)
        outputs = model(seq)

        _, prediction = torch.max(outputs.data, 1)
        test_accuracy += int(torch.sum(prediction == labels.data))
    test_accuracy = test_accuracy / test_count

    print('Epoch: ' + str(epoch) + ' Train Loss: ' + str(train_loss) + ' Train Accuracy: ' + str(
        train_accuracy) + ' Test Accuracy: ' + str(test_accuracy))



