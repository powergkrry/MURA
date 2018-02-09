import torch
import torch.nn as nn
from torch.autograd import Variable
import matplotlib.pyplot as plt
import numpy as np
import os
from PIL import Image
import torchvision
from torchvision.transforms import ToTensor, ToPILImage
import time
import folder2


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

     img_arr = Variable(torch.Tensor(img_arr).view(-1, 1, 512, 350)).cuda()
     return img_arr

torch.manual_seed(1)    # reproducible

EPOCH = 30
BATCH_SIZE = 16 
LR = 0.001     # learning rate
N_TEST_IMG = 5

dir_input = '/hoem04/outofhome/MURA_TRAIN_RESIZE_NOISE/'
dir_output = '/hoem04/outofhome/MURA_TRAIN_RESIZE/'
dir_test_input = '/hoem04/outofhome/MURA_TEST_RESIZE_NOISE/'
dir_test_output = '/hoem04/outofhome/MURA_TEST_RESIZE/'
namelist_input = os.listdir('/hoem04/outofhome/MURA_TRAIN_RESIZE_NOISE/test/')
namelist_output = os.listdir('/hoem04/outofhome/MURA_TRAIN_RESIZE/test/')
namelist_test = os.listdir('/hoem04/outofhome/MURA_TEST_RESIZE/test/')
FILE_NUM = len(namelist_input)

class AutoEncoder(nn.Module):
    def __init__(self):
        super(AutoEncoder, self).__init__()
        """
        self.encoder = nn.Sequential(
            nn.Conv2d(1, 16, 3, stride=1, padding=1),
            nn.BatchNorm2d(16),
            nn.ReLU(),
            nn.Conv2d(16, 32, 3, stride=1, padding=1),
            nn.BatchNorm2d(32),
            nn.ReLU(),
            nn.MaxPool2d(2, stride=2, return_indices=True),

            nn.Conv2d(32, 64, 3, stride=1, padding=1),
            nn.BatchNorm2d(64),
            nn.ReLU(),
#            nn.Conv2d(64, 64, 3, stride=1, padding=1),
#            nn.BatchNorm2d(64),
#            nn.ReLU(),
            nn.MaxPool2d(2, stride=2, return_indices=True),

#            nn.Linear(256*128*87, 1024),
#            nn.BatchNorm2d(1024),
#            nn.ReLU()
	)

        self.decoder = nn.Sequential(
            nn.MaxUnpool2d(2, stride=2),
#            nn.ConvTranspose2d(64, 64, 3, padding=1),
            nn.ConvTranspose2d(64, 32, 3, padding=1),
            
            nn.MaxUnpool2d(2, stride=2),
            nn.ConvTranspose2d(32, 16, 3, padding=1),
            nn.ConvTranspose2d(16, 1, 3, padding=1),

#            nn.Sigmoid()
        )
        """
        self.conv1 = nn.Conv2d(1, 16, 3, stride=1, padding=1)
        self.bn1 = nn.BatchNorm2d(16)
        self.r1 = nn.ReLU()

        self.conv2 = nn.Conv2d(16, 32, 3, stride=1, padding=1)
        self.bn2 = nn.BatchNorm2d(32)
        self.r2 = nn.ReLU()

        self.maxpool1 = nn.MaxPool2d(2, stride=2, return_indices=True)

        self.conv3 = nn.Conv2d(32, 50, 3, stride=1, padding=1)
        self.bn3 = nn.BatchNorm2d(50)
        self.r3 = nn.ReLU()

        self.maxpool2 = nn.MaxPool2d(2, stride=2, return_indices=True)

        self.maxunpool2 = nn.MaxUnpool2d(2, stride=2)

        self.deconv3 = nn.ConvTranspose2d(50, 32, 3, padding=1)
        self.bn4 = nn.BatchNorm2d(32)
        self.r4 = nn.ReLU()

        self.maxunpool1 = nn.MaxUnpool2d(2, stride=2)

        self.deconv2 = nn.ConvTranspose2d(32, 16, 3, padding=1)
        self.bn5 = nn.BatchNorm2d(16)
        self.r5 = nn.ReLU()

        self.deconv1 = nn.ConvTranspose2d(16, 1, 3, padding=1)
#        self.bn6 = nn.BatchNorm2d(1)
#        self.r6 = nn.ReLU()
        
    """
    def forward(self, x):
        encoded = self.encoder(x)
        decoded = self.decoder(encoded)
        return encoded, decoded
    """
    def forward(self, x):
        out = self.r1(self.bn1(self.conv1(x)))
        out = self.r2(self.bn2(self.conv2(out)))

        size1 = out.size()
        out, indices1 = self.maxpool1(out)

        out = self.r3(self.bn3(self.conv3(out)))

        size2 = out.size()
        out, indices2 = self.maxpool2(out)


        out = self.maxunpool2(out, indices2, size2)

        out = self.r4(self.bn4(self.deconv3(out)))

        out = self.maxunpool1(out, indices1, size1)

        out = self.r5(self.bn5(self.deconv2(out)))
        out = self.deconv1(out)

        return(out)


print("generating autoencoder")
autoencoder = AutoEncoder()
autoencoder.cuda()
print(autoencoder)

optimizer = torch.optim.Adam(autoencoder.parameters(), lr=LR, weight_decay=1e-5)
#scheduler = torch.optim.lr_scheduler.StepLR(optimizer, step_size=50, gamma=0.1)

check = torch.load('./pretrained/kang01_epoch30.pth.tar')
autoencoder.load_state_dict(check['model'])
optimizer.load_state_dict(check['state'])

loss_func = nn.MSELoss()

print("loading view_data")
view_data_in = img_to_tensorVariable(
    os.path.join(dir_input,'test/'), namelist_input, 1, N_TEST_IMG
)
view_data_out = img_to_tensorVariable(
    os.path.join(dir_output,'test/'), namelist_output, 1, N_TEST_IMG
)
print("loading complete")

#start_time = time.time()
print("loading train_loader")
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
print("loading complete")

#print("--- %s seconds ---" %(time.time() - start_time))

"""
f, a = plt.subplots(3, N_TEST_IMG, figsize=(5,3))
plt.ion()

for i in range(N_TEST_IMG):
    a[0][i].imshow(view_data_in.data.cpu().numpy()[i].reshape(512, 350), cmap='gray')
    a[0][i].set_xticks(()) 
    a[0][i].set_yticks(())
    a[1][i].imshow(view_data_out.data.cpu().numpy()[i].reshape(512,350), cmap='gray')
    a[1][i].set_xticks(())
    a[1][i].set_yticks(())
"""

for epoch in range(EPOCH):
    train_input_iter = iter(train_loader_input)
    train_output_iter = iter(train_loader_output)

    if len(train_input_iter) != len(train_output_iter):
        print("different number of picture")
        break

    for step in range(len(train_input_iter)):
        data_input, _ = train_input_iter.next()
        data_output, _ = train_output_iter.next()
        data_input = data_input.type(torch.FloatTensor)
        data_output = data_output.type(torch.FloatTensor)

        x = Variable(data_input.cuda())
        y = Variable(data_output.cuda())
        decoded = autoencoder(x) # delete encoded,

        loss = loss_func(decoded, y)
        optimizer.zero_grad()
        loss.backward()
        optimizer.step()

        print("Epoch :", epoch, "| step :",step,"| train loss: %0.6f" % loss.data[0])
        """ 
        if step%100 != 0:
            continue

        decoded_data = autoencoder(view_data_in)
        for i in range(N_TEST_IMG):
            a[2][i].clear()
            a[2][i].imshow(decoded_data.data.cpu().numpy()[i].reshape(512,350), cmap='gray')
            a[2][i].set_xticks(())
            a[2][i].set_yticks(())
        plt.draw()
        plt.pause(0.05)
        """
#    epoch_ = epoch + 1
#    save_name = './pretrained/kang01_epoch' + str(epoch_) + '.pth.tar'
#    torch.save({'model':autoencoder.state_dict(),'state':optimizer.state_dict()}, save_name)
#    scheduler.step()
"""

print("loading test_loader")
folder_test_input = folder2.ImageFolder(
    root = dir_test_input,
    transform = ToTensor(),
    loader = folder2.pil_loader
)
test_loader_input = torch.utils.data.DataLoader(
    folder_test_input, batch_size = 5, shuffle = False)

folder_test_output = folder2.ImageFolder(
    root = dir_test_output,
    transform = ToTensor(),
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

    test_denoise = autoencoder(x)

    f, a = plt.subplots(3, N_TEST_IMG, figsize=(5, 3))

    for i in range(N_TEST_IMG):
        a[0][i].imshow(x.data.cpu().numpy()[i].reshape(512,350), cmap='gray')
        a[0][i].set_xticks(())
        a[0][i].set_yticks(())

    for i in range(N_TEST_IMG):
        a[1][i].imshow(y.data.cpu().numpy()[i].reshape(512,350), cmap='gray')
        a[1][i].set_xticks(())
        a[1][i].set_yticks(())

    for i in range(N_TEST_IMG):
        a[2][i].imshow(test_denoise.data.cpu().numpy()[i].reshape(512,350), cmap='gray')
        a[2][i].set_xticks(())
        a[2][i].set_yticks(())
    plt.show()
"""
