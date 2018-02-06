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
#torch.set_default_tensor_type('torch.cuda.FloatTensor')

EPOCH = 2000
BATCH_SIZE = 32
LR = 0.001     # learning rate
N_TEST_IMG = 5

dir_input = '/hoem04/outofhome/TEMP/'
dir_output = '/hoem04/outofhome/MURA_TRAIN_RESIZE/test/'

namelist_input = os.listdir('/hoem04/outofhome/TEMP/')
namelist_output = os.listdir('/hoem04/outofhome/MURA_TRAIN_RESIZE/test/')

print("first input :", namelist_input[0])
print("first output :", namelist_output[0])

sampleimg_input = Image.open(os.path.join(dir_input, namelist_input[0]))
sampleimg_output = Image.open(os.path.join(dir_output, namelist_input[0]))

fig = plt.figure()
plt.subplot(121)
plt.imshow(sampleimg_input)	# show input image
plt.subplot(122)
plt.imshow(sampleimg_output)	# show output image

#plt.show()

class AutoEncoder(nn.Module):
    def __init__(self):
        super(AutoEncoder, self).__init__()

        self.encoder = nn.Sequential(
            nn.Linear(350*512, 512),
            nn.ReLU(),
            nn.Linear(512, 256),
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
            nn.Linear(256, 512),
            nn.ReLU(),
            nn.Linear(512, 350*512),
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

print("loading view_data_name")
view_data_input = namelist_input[:N_TEST_IMG]
view_data_arr = np.array([])
for name in view_data_input:
    arr=np.array(Image.open(os.path.join(dir_input, name)))
    view_data_arr = np.append(view_data_arr, arr.tolist())

view_data_in = torch.Tensor(view_data_arr)
view_data_in = Variable(view_data_in.view(-1,350*512)/255.).cuda()

view_data_output = namelist_input[:N_TEST_IMG]
view_data_arr = np.array([])
for name in view_data_output:
    arr = np.array(Image.open(os.path.join(dir_output, name)))
    view_data_arr = np.append(view_data_arr, arr.tolist())

view_data_out = torch.Tensor(view_data_arr)
view_data_out = Variable(view_data_out.view(-1,350*512)/255.).cuda()
print("loading complete")

print("loading train_loader_input")
train_loader_in = np.array([])
idx = 0
for name in namelist_input:
    arr = np.array(Image.open(os.path.join(dir_input,name)))
    train_loader_in = np.append(train_loader_in, arr.tolist())
    idx += 1
    if idx%100 == 0 :
        break

train_loader_in = torch.Tensor(train_loader_in)#.cuda()
print(train_loader_in)
train_loader_input = Variable(train_loader_in.view(-1,512*350)/255.).cuda()
print("loading complete")
print(train_loader_input)

print("loading train_loader_output")
train_loader_out = np.array([])
idx = 0
for name in namelist_output:
    arr = np.array(Image.open(os.path.join(dir_output,name)))
    train_loader_out = np.append(train_loader_out, arr.tolist())
    idx += 1
    if idx%100 == 0 :
        break

train_loader_out = torch.Tensor(train_loader_out)#.cuda()
train_loader_output = Variable(train_loader_out.view(-1,512*350)/255.).cuda()
print(train_loader_output)

print("loading complete")
 
print("start learning")
for epoch in range(EPOCH):


    #print(type(torch.Tensor(np.array(inputimg).astype(float))))
    #inputimg = Variable(torch.Tensor(np.array(inputimg).astype(float).reshape(350*512))).cuda()
    #outputimg = Variable(torch.Tensor(np.array(outputimg).astype(float).reshape(350*512))).cuda()


    encoded, decoded = autoencoder(train_loader_input)
#    print("encoded :",encoded)

    loss = loss_func(decoded, train_loader_output)
    optimizer.zero_grad()
    loss.backward()
    optimizer.step()

    print("Epoch :", epoch, "| train loss: %0.4f" % loss.data[0])

    if epoch%100 != 0:
        continue

    print(view_data_in)
    _, decoded_data = autoencoder(view_data_in)
    print(decoded_data)    

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
        a[2][i].imshow(decoded_data.data.cpu().numpy().reshape(-1,512,350)[i], cmap='gray')
        a[2][i].set_xticks(())
        a[2][i].set_yticks(())
    plt.show()
      
