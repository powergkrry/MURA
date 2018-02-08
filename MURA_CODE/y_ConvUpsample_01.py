import torch
import torch.nn as nn
import torch.nn.functional as F
from torch.autograd import Variable
import matplotlib.pyplot as plt
import numpy as np
import os
from PIL import Image
import torchvision
from torchvision.transforms import ToTensor, ToPILImage
import time
import folder2
import resnet2

def img_to_tensorVariable(dir_path, name_list, num_img_from, num_img_to):
     img_list = []

     for i, name in enumerate(name_list):
          if i + 1 < num_img_from:
               continue

          img = Image.open(os.path.join(dir_path, name))
          img_list.append(list(img.getdata()))

          if i + 1 == num_img_to:
               img_arr = np.asfarray(img_list)/255.
               break

#          if i % 1000 == 0:
#               print(i)

     img_arr = Variable(torch.Tensor(img_arr).view(-1, 1, 512, 350).cuda())
#     print("for done")
#     img_arr = torch.Tensor(img_arr).view(-1, 512*350)
#     print("to return")
     return img_arr

torch.manual_seed(1)    # reproducible
#torch.set_default_tensor_type('torch.cuda.FloatTensor')

EPOCH = 180
BATCH_SIZE = 16
LR = 0.001     # learning rate
N_TEST_IMG = 5

dir_input = '/hoem04/outofhome/MURA_TRAIN_RESIZE_NOISE/'
dir_output = '/hoem04/outofhome/MURA_TRAIN_RESIZE/'
namelist_input = os.listdir('/hoem04/outofhome/MURA_TRAIN_RESIZE_NOISE/test/')
namelist_output = os.listdir('/hoem04/outofhome/MURA_TRAIN_RESIZE/test/')

class AutoEncoder(nn.Module):
    def __init__(self):
        super(AutoEncoder, self).__init__()
        
        self.conv1 = nn.Conv2d(1, 32, 3, stride=1, padding=1)
        self.bn1 = nn.BatchNorm2d(32)
        self.conv2 = nn.Conv2d(32, 64, 3, stride=1, padding=1)
        self.bn2 = nn.BatchNorm2d(64)
        self.conv3 = nn.Conv2d(64, 32, 3, stride=1, padding=1)
        self.bn3 = nn.BatchNorm2d(32)
        self.upsample1 = nn.Upsample(scale_factor=2)
        self.conv4 = nn.Conv2d(32, 1, 3, stride=1, padding=1)

    def forward(self, x):
        x = self.conv1(x)
        x = self.bn1(x)
        x = F.max_pool2d(F.relu(x), 2)
        x = self.conv2(x)
        x = self.bn2(x)
        x = F.relu(x)
        x = self.conv3(x)
        x = self.bn3(x)
        x = self.upsample1(F.relu(x))
        x = F.relu(self.conv4(x))

        return x

    def num_flat_features(self, x):
        size = x.size()[1:]
        num_features = 1
        for s in size:
            num_features *= s
        return num_features

print("generating autoencoder")
autoencoder = AutoEncoder()
autoencoder.cuda()
print(autoencoder)

optimizer = torch.optim.Adam(autoencoder.parameters(), lr=LR)
loss_func = nn.MSELoss()

print("loading view_data")
view_data_in = img_to_tensorVariable(
    os.path.join(dir_input,'test/'), namelist_input, 1, N_TEST_IMG
)
view_data_out = img_to_tensorVariable(
    os.path.join(dir_output,'test/'), namelist_output, 1, N_TEST_IMG
)
print("loading complete")

start_time = time.time()
print("loading train_loader")
#train_loader_input = img_to_tensorVariable(dir_input, namelist_input, 1, 10000)
folder_input = folder2.ImageFolder(
    root = dir_input,
    transform = ToTensor(),
    loader = folder2.pil_loader
)
train_loader_input = torch.utils.data.DataLoader(
    folder_input, batch_size = BATCH_SIZE, shuffle = False)

folder_output = folder2.ImageFolder(
    root = dir_output,
    transform = ToTensor(),
    loader = folder2.pil_loader
)
train_loader_output = torch.utils.data.DataLoader(
    folder_output, batch_size = BATCH_SIZE, shuffle = False)

#train_loader_output = img_to_tensorVariable(dir_output, namelist_output, 1, 100)
print("loading complete")
print("--- %s seconds ---" %(time.time() - start_time))

print("start learning")

f, a = plt.subplots(3, N_TEST_IMG, figsize=(5,3))
plt.ion()

for i in range(N_TEST_IMG):
    a[0][i].imshow(view_data_in.data.cpu().numpy()[i].reshape(512, 350), cmap='gray')
    a[0][i].set_xticks(()) 
    a[0][i].set_yticks(())
    a[1][i].imshow(view_data_out.data.cpu().numpy()[i].reshape(512,350), cmap='gray')
    a[1][i].set_xticks(())
    a[1][i].set_yticks(())

for epoch in range(EPOCH):
    
    for step, (data_input, data_output) in enumerate(zip(train_loader_input, train_loader_output)):

        x = Variable(data_input[0].cuda())
        y = Variable(data_output[0].cuda())
        decoded = autoencoder(x)

        loss = loss_func(decoded, y)
        optimizer.zero_grad()
        loss.backward()
        optimizer.step()

        print("Epoch :", epoch, "| step :",step,"| train loss: %0.6f" % loss.data[    0])

        if step%50 != 0:
            continue

        decoded_data = autoencoder(view_data_in)

        for i in range(N_TEST_IMG):
            a[2][i].clear()
            a[2][i].imshow(decoded_data.data.cpu().numpy()[i].reshape(512,350), cmap='gray')
            a[2][i].set_xticks(())
            a[2][i].set_yticks(())
        plt.draw()
        plt.pause(0.05)

#torch.save(autoencoder.state_dict(), './06_epoch180.pt')
torch.save({'model':autoencoder.state_dict(),'state':optimizer.state_dict()}, './07_epoch180.pth.tar')


_, test_denoise = autoencoder(test_loader_input)

f, a = plt.subplots(3, N_TEST_IMG, figsize=(5, 3))

for i in range(N_TEST_IMG):
    a[0][i].imshow(test_loader_input.data.cpu().numpy()[i].reshape(512,350), cmap='gray')
    a[0][i].set_xticks(())
    a[0][i].set_yticks(())

for i in range(N_TEST_IMG):
    a[1][i].imshow(test_loader_output.data.cpu().numpy()[i].reshape(512,350), cmap='gray')
    a[1][i].set_xticks(())
    a[1][i].set_yticks(())

for i in range(N_TEST_IMG):
    a[2][i].imshow(test_denoise.data.cpu().numpy()[i].reshape(512,350), cmap='gray')
    a[2][i].set_xticks(())
    a[2][i].set_yticks(())
plt.show()
