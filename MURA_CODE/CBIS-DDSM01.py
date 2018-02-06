import torch
import torch.nn as nn
import torch.nn.functional as F
import torch.optim as optim
import matplotlib.pyplot as plt
import numpy as np
import time
from torchvision import datasets, transforms
from torch.autograd import Variable
from PIL import Image
import folder2

class MnistModel(nn.Module):
    def __init__(self):
        super(MnistModel, self).__init__()
        # input is 150x150
        self.conv1 = nn.Conv2d(3, 32, 5, stride=2, padding=2) # feature map size is 75x75
        self.conv1_bn = nn.BatchNorm2d(32)
        # input is 73x73
        self.conv2 = nn.Conv2d(32, 256, 5, stride=2, padding=2) # feature map size is 37x37
        self.conv2_bn = nn.BatchNorm2d(256)
        # input is 37x37
        self.conv3 = nn.Conv2d(256, 64, 5, stride=2, padding=2) # feature map size is 19x19
        self.conv3_bn = nn.BatchNorm2d(64)
        # input is 19x19
        self.conv4 = nn.Conv2d(64, 50, 3, padding=1) # feature map size is 19x19
        self.conv4_bn = nn.BatchNorm2d(50)

        # feature map size is 15x15
        self.fc1 = nn.Linear(50 * 17 * 17, 2048)
        self.fc1_bn = nn.BatchNorm2d(2048)
        self.fc2 = nn.Linear(2048, 2048)
        self.fc2_bn = nn.BatchNorm2d(2048)
        self.fc3 = nn.Linear(2048, 512)
        self.fc3_bn = nn.BatchNorm2d(512)
        self.fc4 = nn.Linear(512, 2)

    def forward(self, x):
        x = F.max_pool2d(F.relu(self.conv1_bn(self.conv1(x))), 3, stride=1) # 73x73
        x = F.relu(self.conv2_bn(self.conv2(x))) # 37x37
        x = F.relu(self.conv3_bn(self.conv3(x))) # 19x19
        x = F.max_pool2d(F.relu(self.conv4_bn(self.conv4(x))), 3, stride=1) # 17x17

        x = x.view(-1, 50 * 17 * 17)  # reshape Variable
        x = F.relu(self.fc1_bn(self.fc1(x)))
#        x = F.dropout(x, training=self.training)
        x = F.relu(self.fc2_bn(self.fc2(x)))
#        x = F.dropout(x, training=self.training)
        x = F.relu(self.fc3_bn(self.fc3(x)))
#        x = F.dropout(x, training=self.training)
        x = self.fc4(x)
        return F.log_softmax(x)

start = time.time()

model = MnistModel()

##model.load_state_dict(torch.load('./trained/30_epoch400.pt'))
model.cuda()

for p in model.parameters():
    print(p.size())

optimizer = optim.Adam(model.parameters(), lr=0.001)

data_transform = transforms.Compose([
        transforms.RandomSizedCrop(150),
        #transforms.RandomHorizontalFlip(),
        transforms.ToTensor()
    ])

model.train()
test_accu = []
i = 0
for epoch in range(10000):
    ddsm_dataset = folder2.ImageFolder(
        root='/hoem04/powergkrry/CBIS-DDSM_data/Test1data/Test1data_PNG',
        transform = data_transform, 
        loader = folder2.pil_loader )

    train_loader = torch.utils.data.DataLoader(
        ddsm_dataset, batch_size=8, shuffle=True)

    print("dataset :")
    print(ddsm_dataset)
    
    for data, target in train_loader:
        data, target = Variable(data).cuda(), Variable(target).cuda()
        #print(data)
        #print(target)
        optimizer.zero_grad()
        output = model(data)
        loss = F.nll_loss(output, target)
        loss.backward()  # calc gradients
        optimizer.step()  # update gradients
        prediction = output.data.max(1)[1]  # first column has actual prob.
        accuracy = prediction.eq(target.data).sum() / 8 * 100
        if i % 400 == 0:
            print('Train Step: {}\tLoss: {:.3f}\tAccuracy: {:.3f}'.format(i, loss.data[0], accuracy))
            """
            model.eval()
            correct = 0
            for data, target in test_loader:
                data, target = Variable(data, volatile=True).cuda(), Variable(target)
                output = model(data)
                prediction = output.data.max(1)[1]
                correct += prediction.cpu().eq(target.data).sum()
                accuracytest = 100. * correct / len(test_loader.dataset)
                test_accu.append(accuracytest)

            print('Test set: Accuracy: {:.2f}%\n'.format(accuracytest))
            """
        i += 1

print('Train Step: {}\tLoss: {:.3f}\tAccuracy: {:.3f}'.format(i, loss.data[0], accuracy))
"""
model.eval()
correct = 0
for data, target in test_loader:
    data, target = Variable(data, volatile=True).cuda(), Variable(target)
    output = model(data)
    prediction = output.data.max(1)[1]
    correct += prediction.cpu().eq(target.data).sum()
    accuracytest = 100. * correct / len(test_loader.dataset)
    test_accu.append(accuracytest)

print('Test set: Accuracy: {:.2f}%\n'.format(accuracytest))
"""
#print(test_accu)

#torch.save(model.state_dict(), './trained/30_epoch500.pt')

end = time.time() - start
print(end)
