# just a comment
import numpy as np
import pandas as pd
from collections import Counter
import matplotlib.pyplot as plt
import torch
import torch.nn as nn
from torch.utils.data import DataLoader,WeightedRandomSampler,random_split


def get_loader(DataSets):

    # oversampling the unbalanced data
    class_weights = [3,1] # the balance ratio is 1 to 5
    sample_weights = [0]*len(DataSets)

    for idx, (data,label) in enumerate(DataSets):
        if label>0:
         label = 1
        class_weight = class_weights[int(label)]
        sample_weights[idx] = class_weight

    sampler = WeightedRandomSampler(sample_weights, num_samples = len(sample_weights), replacement=True)

    # Dataloader
    train_loader = DataLoader(DataSets,
        batch_size=40,sampler = sampler)

    return train_loader



data = pd.read_csv('/Users/zvistein/Downloads/mafat_wifi_challenge_training_set_v1.csv')
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
mn = np.min(rss[int(l/wind):,:]+rss[0:int(l/wind),:])
mx = np.max(rss[int(l/wind):,:]+rss[0:int(l/wind),:])
rss_n[:,:,3] = (rss[int(l/wind):,:]+rss[0:int(l/wind),:]-mn)/(mx-mn)
mn = np.min(rss[int(l/wind):,:]-rss[0:int(l/wind),:])
mx = np.max(rss[int(l/wind):,:]-rss[0:int(l/wind),:])
rss_n[:,:,2] = (rss[int(l/wind):,:]-rss[0:int(l/wind),:]-mn)/(mx-mn)


b = data.Num_People[0:l]
num = b.values.reshape(int(l/360),360)
gt = np.zeros(int(l/360))
for i in np.arange(0,int(l/360)):
 n = num[i,:]
 b = Counter(n)
 gt[i] = np.sign(b.most_common(1)[0][0])

train_data = []
for i in range(len(rss_n)):
   train_data.append([rss_n[i], gt[i]])


class FN(nn.Module):
    def __init__(self, num_classes = 4):
        super(FN, self).__init__()

        # Output size after convolution filter
        # ((w-f+2P)/s) +1
        self.conv1 = nn.Conv2d(in_channels=1, out_channels=12, kernel_size=[1,2], stride=1, padding=0)
        # Shape= (b_s,12,50,50)
        self.bn1 = nn.BatchNorm2d(num_features=12)
        # Shape= (b_s,12,50,50)

        # Input shape= (b_s,1,50,50)

        self.fc1 = nn.Linear(in_features=4*360, out_features=36000)
        self.bn2 = nn.BatchNorm1d(36000)
        self.fc2 = nn.Linear(in_features=36000, out_features = 1000)
        self.bn3 = nn.BatchNorm1d(1000)
        self.fc3 = nn.Linear(in_features=1000, out_features = 100)
        self.bn4 = nn.BatchNorm1d(100)
        self.fc4 = nn.Linear(in_features=100, out_features = 50)
        self.bn5 = nn.BatchNorm1d(50)
        self.fc5 = nn.Linear(in_features=50, out_features = 25)
        self.bn6 = nn.BatchNorm1d(25)
        self.fc6 = nn.Linear(in_features=25, out_features = 10)
        self.bn7 = nn.BatchNorm1d(10)
        self.fc7 = nn.Linear(in_features=10, out_features = 1)
        self.bn8 = nn.BatchNorm1d(1)
        self.sm  = nn.Softmax()

        self.relu = nn.ReLU()
        self.Lrelu = nn.LeakyReLU()

        # Feed forwad function

    def forward(self, input):
        # tmpInput = input
        # output = self.conv1(input)
        # output = self.bn1(output)
        # output = self.Lrelu(output)

        output = input.view(-1, 4*360)
        # input  = input.view(-1, 360 * 2)
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
        output = self.relu(output)
        output = self.bn8(output)
        output = self.sm(output)


        return output


test_count  = len(rss_n)//7
train_count = len(rss_n)-test_count

train_sets, test_setes = random_split(train_data,[train_count,test_count])

test_loader = DataLoader(test_setes,
    batch_size=500, shuffle=True)
train_loader = get_loader(train_sets)
# train_loader = DataLoader(train_sets,
#     batch_size=250, shuffle=True)

torch.manual_seed(3407)
model = FN(num_classes=1)

optimizer     = torch.optim.Adam(model.parameters(), lr=0.1)
scheduler = torch.optim.lr_scheduler.StepLR(optimizer, step_size=15, gamma=(0.9))

loss_function = nn.BCELoss()

num_epochs = 100
s_loss = torch.zeros(num_epochs*len(train_loader))
idx = 0

for epoch in range(num_epochs):

    # Evaluation and training on training dataset
    model.train()
    train_accuracy = 0.0
    train_loss = 0.0

    for i, (seq, labels) in enumerate(train_loader):
        labels[labels > 0] = 1
        optimizer.zero_grad()
        seq = seq[None, :]
        seq = seq.permute(1, 0, 2, 3)
        outputs = model(seq.float())
        loss = loss_function(outputs.view(-1), labels.float())        # print(loss)
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
        labels[labels > 0] = 1
        seq = seq[None, :]
        seq = seq.permute(1, 0, 2, 3)

        outputs = model(seq.float())

        _, prediction = torch.max(outputs.data, 1)
        test_accuracy += int(torch.sum(prediction == labels.data))
    test_accuracy = test_accuracy / test_count

    print('Epoch: ' + str(epoch) + ' Train Loss: ' + str(train_loss) + ' Train Accuracy: ' + str(
        train_accuracy) + ' Test Accuracy: ' + str(test_accuracy))