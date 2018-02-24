"""
1. copy & paste the network you want to see test images
2. set the pretrained file
"""

import torch
import torch.nn as nn
from torch.autograd import Variable
import matplotlib.pyplot as plt
import numpy as np
from torchvision import transforms
import folder2
import resnet2

class AutoEncoder(nn.Module):
    def __init__(self):
        super(AutoEncoder, self).__init__()

        self.conv1 = nn.Conv2d(1, 64, 3, stride=1, padding=1)
        self.bn1 = nn.BatchNorm2d(64)
        self.conv2 = nn.Conv2d(64, 64, 3, stride=1, padding=1)
        self.conv3 = nn.Conv2d(64, 1, 3, stride=1, padding=1)
        self.ReLU = nn.ReLU()

    def forward(self, x):
        residual = x
        out = self.ReLU(self.bn1(self.conv1(x)))
        out = self.ReLU(self.bn1(self.conv2(out)))
        out = self.ReLU(self.bn1(self.conv2(out)))
        out = self.ReLU(self.bn1(self.conv2(out)))
        out = self.ReLU(self.bn1(self.conv2(out)))
        out = self.ReLU(self.bn1(self.conv2(out)))
        out = self.ReLU(self.bn1(self.conv2(out)))
        out = self.conv3(out)
        out += residual
        out = self.ReLU(out)
        return out

LR2 = 0.0001
N_TEST_IMG = 5

dir_test_input = '/home/powergkrry/MURA/MURA_TEST_RESIZE_NOISE/'
dir_test_output = '/home/powergkrry/MURA/MURA_TEST_RESIZE/'

autoencoder = AutoEncoder()
autoencoder = torch.nn.DataParallel(autoencoder, device_ids = [0, 1])
autoencoder.cuda()
optimizer = torch.optim.Adam(autoencoder.parameters(), lr=LR2, weight_decay=1e-5)

autoencoder.load_state_dict(torch.load('./pretrained/y_ResNet01_epoch30_model.pth.tar'))
optimizer.load_state_dict(torch.load('./pretrained/y_ResNet01_epoch30_optimizer.pth.tar'))

trans_comp = transforms.Compose([
        transforms.CenterCrop(100),
        transforms.ToTensor()
    ])

print("loading test_loader")
folder_test_input = folder2.ImageFolder(
    root = dir_test_input,
    transform = trans_comp,
    loader = folder2.pil_loader
)
test_loader_input = torch.utils.data.DataLoader(
    folder_test_input, batch_size = 5, shuffle = False)

folder_test_output = folder2.ImageFolder(
    root = dir_test_output,
    transform = trans_comp,
    loader = folder2.pil_loader
)
test_loader_output = torch.utils.data.DataLoader(
    folder_test_output, batch_size = 5, shuffle = False)
print("loading complete")


test_input_iter = iter(test_loader_input)
test_output_iter = iter(test_loader_output)

for i in range(len(test_input_iter)):
    data_input, _ = test_input_iter.next()
    data_output, _ = test_output_iter.next()
    data_input = data_input.type(torch.FloatTensor)
    data_output = data_output.type(torch.FloatTensor)

    x = Variable(data_input.cuda())
    y = Variable(data_output.cuda())

    test_denoise = autoencoder(y)

    f, a = plt.subplots(3, N_TEST_IMG, figsize=(10, 6))

    for i in range(N_TEST_IMG):
        a[0][i].imshow(x.data.cpu().numpy()[i].reshape(100,100), cmap='gray')
        a[0][i].set_xticks(())
        a[0][i].set_yticks(())

    for i in range(N_TEST_IMG):
        a[1][i].imshow(y.data.cpu().numpy()[i].reshape(100,100), cmap='gray')
        a[1][i].set_xticks(())
        a[1][i].set_yticks(())

    for i in range(N_TEST_IMG):
        a[2][i].imshow(test_denoise.data.cpu().numpy()[i].reshape(100,100), cmap='gray')
        a[2][i].set_xticks(())
        a[2][i].set_yticks(())
    plt.show()



