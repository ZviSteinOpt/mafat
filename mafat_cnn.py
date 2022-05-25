# just a comment
import numpy as np
import pandas as pd
from collections import Counter
import matplotlib.pyplot as plt
import torch
import torch.nn as nn
from torch.utils.data import DataLoader,WeightedRandomSampler,random_split
import time

if torch.cuda.is_available():
    device = torch.device('cuda')
elif torch.backends.mps.is_available():
    device = torch.device('cpu')
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
        batch_size=50,sampler = sampler)

    return train_loader


data = pd.read_csv('C:/Users/aviel/Desktop/MAFAT/mafat_wifi_challenge_training_set_v1.csv')
# data = pd.read_csv('/Users/zvistein/Downloads/mafat_wifi_challenge_training_set_v1.csv')
all_data   = []
all_labels = []
weit       = [0,0]
for ii in range(1):
    wind = 360
    rm = 2
    Dr = data.RSSI_Right[data.Room_Num == rm]
    Dl = data.RSSI_Left[data.Room_Num == rm]
    La = data.Num_People[data.Room_Num == rm]
    l = (len(Dr)-len(Dr)%wind)
    b =  Dr[(ii*30):(l+(ii*30))]
    a =  Dl[(ii*30):(l+(ii*30))]
    a = np.append(a,b)
    rss =  a.reshape(2,len(a)//2)
    rss_n = np.zeros([1463,720])
    for ij in range(1463):
        rss_n[ij,0:360] = rss[0,(360*ij):(360*(ij+1))]
        rss_n[ij,360:] = rss[1,(360*ij):(360*(ij+1))]

    stdz = []
    for i in range(len(rss_n)):

        mn = np.mean(rss_n[i,:])
        mx0 = np.std(rss_n[i,:])
        rss_n[i,:] = (rss_n[i,:] - mn) / mx0
        if mx0==0:
            stdz.append(i)

    rss_n = np.delete(rss_n,stdz,0)


    b = La[(ii*30):(l+(ii*30))]
    num = b.values.reshape(int(l/360),360)
    gt = np.zeros(int(l/360))
    for i in np.arange(0,int(l/360)):
     n = num[i,:]
     b = Counter(n)
     gt[i] = np.sign(b.most_common(1)[0][0])

    all_data.append(rss_n)
    all_labels.append(gt)
    classs = 2
    for j in range(classs):
     weit[j] = weit[j]+(1/classs)/(sum(gt==j)/len(gt))


weit = np.round(weit/min(weit))
train_data = []
train_data_of = []
flag = np.zeros(4)
count = 0
count_1 = 0


for ii in range(1):
    rss_n = all_data[ii]
    gt    = all_labels[ii]
    for i in range(len(rss_n)):
       train_data.append([rss_n[i], gt[i]])
       count_1 = count_1 + 1
       flag[int(gt[i])] = flag[int(gt[i])]+1
       if flag[int(gt[i])]<100:
           train_data_of.append([rss_n[i], gt[i]])
           count = count+1


class FN(nn.Module):
    def __init__(self, num_classes = 4):
        super(FN, self).__init__()

        # Output size after convolution filter
        # ((w-f+2P)/s) +1
        self.conv1 = nn.Conv1d(in_channels=1, out_channels=40, kernel_size=[5], stride=3, padding=0)
        self.bn1 = nn.BatchNorm1d(num_features=40)
        self.conv2 = nn.Conv1d(in_channels=40, out_channels=80, kernel_size=[10], stride=3, padding=0)
        self.bn11 = nn.BatchNorm1d(num_features=80)
        self.conv3 = nn.Conv1d(in_channels=80, out_channels=160, kernel_size=[20], stride=3, padding=0)
        self.bn12 = nn.BatchNorm1d(num_features=160)


        # Shape= (b_s,12,50,50)
        # Shape= (b_s,12,50,50)

        # Input shape= (b_s,1,50,50)

        self.fc1 = nn.Linear(in_features=20*160, out_features=2000)
        self.bn2 = nn.BatchNorm1d(2000)
        self.fc2 = nn.Linear(in_features=2000, out_features = 200)
        self.bn3 = nn.BatchNorm1d(200)
        self.fc3 = nn.Linear(in_features=200, out_features = 20)
        self.bn4 = nn.BatchNorm1d(20)
        self.fc5 = nn.Linear(in_features=20, out_features = 2)
        self.DO  = nn.Dropout(p=0.5)
        self.relu = nn.ReLU()
        self.Lrelu = nn.LeakyReLU()

        # Feed forwad function

    def forward(self, input):
        output = self.conv1(input)
        output = self.bn1(output)
        output = self.Lrelu(output)
        output = self.conv2(output)
        output = self.bn11(output)
        output = self.Lrelu(output)
        output = self.conv3(output)
        output = self.bn12(output)
        output = self.Lrelu(output)
        output = output.view(-1, 160*20)
        output = self.fc1(output)
        output = self.bn2(output)
        output = self.Lrelu(output)
        output = self.fc2(output)
        output = self.bn3(output)
        output = self.DO(output)
        output = self.Lrelu(output)
        output = self.fc3(output)
        output = self.bn4(output)
        output = self.Lrelu(output)
        output = self.DO(output)
        output = self.fc5(output)
        return output


test_count  = count_1//20
train_count = count_1-test_count
# train_count = np.floor(count*0.8)
# test_count  = count-np.floor(count*0.8)

train_sets, test_setes = random_split(train_data,[train_count,test_count])
# train_sets, test_setes = random_split(train_data_of,[int(train_count),int(test_count)])
test_loader = DataLoader(test_setes,
    batch_size=500, shuffle=True)
train_loader = get_loader(train_sets,weit)
# train_loader = DataLoader(train_sets,
#     batch_size=250, shuffle=True)
train_sets = get_loader(train_sets,weit)

torch.manual_seed(1)
model = FN(num_classes=1).to(device)

optimizer     = torch.optim.SGD(model.parameters(), lr=0.001,weight_decay=1e-5)
scheduler = torch.optim.lr_scheduler.StepLR(optimizer, step_size=800, gamma=0.5)

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
        seq = seq.permute(1, 0,2)
        seq = seq.float()
        seq = seq.to(device)
        outputs = model(seq)
        labels = labels.long()
        labels = labels.to(device)
        loss = loss_function(outputs, labels)# + 0.05*torch.sqrt(sum(sum(outputs*outputs)))
        idx = idx+1
        loss.backward()
        optimizer.step()
        #scheduler.step()
        train_loss += loss.cpu().data * seq.size(0)
        _, prediction = torch.max(outputs.data, 1)

        train_accuracy += int(torch.sum(prediction == labels.data))

    train_accuracy = train_accuracy / train_count
    train_loss = train_loss / train_count
    # Evaluation on testing dataset
    model.eval()

    test_accuracy = 0.0
    test_loss     = 0.0
    for i, (seq, labels) in enumerate(test_loader):
        seq = seq[None, :]
        seq = seq.permute(1, 0,2)
        seq = seq.float()
        seq = seq.to(device)
        labels = labels.long()
        labels = labels.to(device)
        outputs = model(seq)
        loss = loss_function(outputs, labels)
        test_loss += loss.cpu().data * seq.size(0)

        _, prediction = torch.max(outputs.data, 1)
        test_accuracy += int(torch.sum(prediction == labels.data))

    test_accuracy = test_accuracy / test_count
    test_loss     = test_loss / test_count

    print('Epoch: ' + str(epoch) + ' Train Loss: ' + str(train_loss) + ' Train Accuracy: ' + str(
        train_accuracy) +' Test Loss: ' + str(test_loss) + ' Test Accuracy: ' + str(test_accuracy))



