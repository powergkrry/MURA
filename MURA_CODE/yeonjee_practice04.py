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

          if i % 1000 == 0:
               print(i)

#     img_arr = Variable(torch.Tensor(img_arr).view(-1, 512*350)).cuda()
     print("for done")
     img_arr = torch.Tensor(img_arr).view(-1, 512*350)
     print("to return")
     return img_arr


torch.manual_seed(1)    # reproducible
#torch.set_default_tensor_type('torch.cuda.FloatTensor')

EPOCH = 2000
BATCH_SIZE = 32
LR = 0.001     # learning rate
N_TEST_IMG = 5

dir_input = '/hoem04/outofhome/TEMP/'
dir_output = '/hoem04/outofhome/MURA_TRAIN_RESIZE/test/'
namelist_input = os.listdir('/hoem04/outofhome/TEMP/')
namelist_output = os.listdir('/hoem04/outofhome/MURA_TRAIN_RESIZE/test/')
"""
class AutoEncoder(nn.Module):
    def __init__(self):
        super(AutoEncoder, self).__init__()

        self.encoder = nn.Sequential(
            nn.Linear(350*512, 700),
            nn.ReLU(),
            nn.Linear(700, 256),
            nn.ReLU(),
            nn.Linear(256, 32),
            nn.ReLU(),
            #nn.Linear(1024, 128),
            #nn.ReLU(),
            #nn.Linear(128, 16)
	)

        self.decoder = nn.Sequential(
            #nn.Linear(16, 128),
            #nn.ReLU(),
            #nn.Linear(128, 1024),
            #nn.ReLU(),
            nn.Linear(32, 256),
            nn.ReLU(),
            nn.Linear(256, 700),
            nn.ReLU(),
            nn.Linear(700, 350*512),
            nn.Sigmoid()
        )

    def forward(self, x):
        encoded = self.encoder(x)
        decoded = self.decoder(encoded)
        return encoded, decoded

print("generating autoencoder")
autoencoder = AutoEncoder()
autoencoder.cuda()
print(autoencoder)

optimizer = torch.optim.Adam(autoencoder.parameters(), lr=LR)
loss_func = nn.MSELoss()

print("loading view_data")
view_data_in = img_to_tensorVariable(dir_input, namelist_input, 1, N_TEST_IMG)
view_data_out = img_to_tensorVariable(dir_output, namelist_output, 1, N_TEST_IMG)
print("loading complete")
"""
start_time = time.time()
print("loading train_loader")
train_loader_input = img_to_tensorVariable(dir_input, namelist_input, 1, 10000)
#train_loader_output = img_to_tensorVariable(dir_output, namelist_output, 1, 100)
print("loading complete")
print("--- %s seconds ---" %(time.time() - start_time))

"""
print("loading test_loader")
test_loader_input = img_to_tensorVariable(dir_input, namelist_input, 101, 105)
test_loader_output = img_to_tensorVariable(dir_output, namelist_output, 101, 105)

print("start learning")
for epoch in range(EPOCH):
    encoded, decoded = autoencoder(train_loader_input)

    loss = loss_func(decoded, train_loader_output)
    optimizer.zero_grad()
    loss.backward()
    optimizer.step()

    print("Epoch :", epoch, "| train loss: %0.6f" % loss.data[0])

    if epoch%20000 != 0:
        continue

    _, decoded_data = autoencoder(view_data_in)

    f, a = plt.subplots(3, N_TEST_IMG, figsize=(5, 3))

    for i in range(N_TEST_IMG):
        a[0][i].imshow(view_data_in.data.cpu().numpy()[i].reshape(512,350), cmap='gray')
        a[0][i].set_xticks(())
        a[0][i].set_yticks(())

    for i in range(N_TEST_IMG):
        a[1][i].imshow(view_data_out.data.cpu().numpy()[i].reshape(512,350), cmap='gray')
        a[1][i].set_xticks(())
        a[1][i].set_yticks(())

    for i in range(N_TEST_IMG):
        a[2][i].imshow(decoded_data.data.cpu().numpy()[i].reshape(512,350), cmap='gray')
        a[2][i].set_xticks(())
        a[2][i].set_yticks(())
    plt.show()


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
"""
