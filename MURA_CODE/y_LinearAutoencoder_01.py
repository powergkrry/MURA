import torch
import torch.nn as nn
from torch.autograd import Variable
import matplotlib.pyplot as plt
import numpy as np
import os
from PIL import Image
import torchvision
from torchvision.transforms import ToTensor, ToPILImage

torch.manual_seed(1)    # reproducible

EPOCH = 10
BATCH_SIZE = 32
LR = 0.0005     # learning rate
N_TEST_IMG = 5

dir_input = '/hoem04/outofhome/TEMP/'
dir_output = '/hoem04/outofhome/MURA_TRAIN_RESIZE/test/'

namelist_input = os.listdir('/hoem04/outofhome/TEMP/')
namelist_output = os.listdir('/hoem04/outofhome/MURA_TRAIN_RESIZE/test/')

print("name input :", namelist_input[0])
print("name output :", namelist_output[0])

sampleimg_input = Image.open(os.path.join(dir_input, namelist_input[0]))
sampleimg_output = Image.open(os.path.join(dir_output, namelist_input[0]))

fig = plt.figure()
plt.subplot(121)
plt.imshow(sampleimg_input)	# show input image
plt.subplot(122)
plt.imshow(sampleimg_output)	# show output image

plt.show()

class AutoEncoder(nn.Module):
    def __init__(self):
        super(AutoEncoder, self).__init__()

        self.encoder = nn.Sequential(
            nn.Linear(350*512, 2048),
            nn.ReLU(),
            nn.Linear(2048, 1024),
            nn.ReLU(),
            nn.Linear(1024, 128),
            nn.ReLU(),
            nn.Linear(128, 16)
	)

        self.decoder = nn.Sequential(
            nn.Linear(16, 128),
            nn.ReLU(),
            nn.Linear(128, 1024),
            nn.ReLU(),
            nn.Linear(1024, 2048),
            nn.ReLU(),
            nn.Linear(2048, 350*512)
        )

    def forward(self, x):
        encoded = self.encoder(x)
        decoded = self.decoder(encoded)
        return encoded, decoded

autoencoder = AutoEncoder()
autoencoder.cuda()
print(autoencoder)

optimizer = torch.optim.Adam(autoencoder.parameters(), lr=LR)
loss_func = nn.MSELoss()

view_data_name = namelist_input[:N_TEST_IMG]
view_data_arr = np.array([])
for name in view_data_name:
    arr=np.array(torch.load(os.path.join(dir_input, name)))
    view_data_arr = np.append(view_data_arr, arr.tolist())

print(type(view_data_arr))
print(view_data_arr)

view_data = Variable(view_data_arr.view(-1, 350*512).type(torch.FloatTensor)/255.).cuda()

for epoch in range(EPOCH):
    for step, inputname in enumerate(namelist_input):
        inputimg = Image.open(os.path.join(dir_input, inputname))
        outputimg = Image.open(os.path.join(dir_output, inputname))

        inputimg = Variable(ToTensor(inputimg)).cuda()
        outputimg = Variable(ToTensor(outputimg)).cuda()
        
        encoded, decoded = autoencoder(inputimg)

        loss = loss_func(decoded, outputimg)
        optimizer.zero_grad()
        loss.backward()
        optimizer.step()

        if step % 500 == 0 and epoch in [0, 5,EPOCH-1]:
            print("Epoch :", epoch, "| train loss: 0.4f" % loss.data[0])

            for i in view_data:
                _, data = autoencoder(i)
                decoded_data.append(data)

            f, a = plt.subplots(2, N_TEST_IMG, figsize=(5, 2))

            for i in range(N_TEST_IMG):
                a[0][i].imshow(np.reshape(view_data.data.cpu().numpy()[i], (28, 28)), cmap='gray')
                a[0][i].set_xticks(())
                a[0][i].set_yticks(())

            for i in range(N_TEST_IMG):
                a[1][i].clear()
                a[1][i].imshow(ToPILImage(decoded_data[i].cpu()), cmap='gray')
                a[1][i].set_xticks(())
                a[1][i].set_yticks(())
            plt.show()
      
