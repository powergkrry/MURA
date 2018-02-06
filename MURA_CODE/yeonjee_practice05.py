import torch
import torch.nn as nn
from torch.autograd import Variable
import matplotlib.pyplot as plt
import numpy as np
import os
from PIL import Image
import torchvision
from torchvision import transforms
import time
import folder2
import png

"""
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

"""
torch.manual_seed(1)    # reproducible
#torch.set_default_tensor_type('torch.cuda.FloatTensor')

EPOCH = 2000
BATCH_SIZE = 32
LR = 0.001     # learning rate
N_TEST_IMG = 5

"""
dir_input = '/hoem04/outofhome/TEMP/'
dir_output = '/hoem04/outofhome/MURA_TRAIN_RESIZE/test/'
namelist_input = os.listdir('/hoem04/outofhome/TEMP/')
namelist_output = os.listdir('/hoem04/outofhome/MURA_TRAIN_RESIZE/test/')
"""
"""
data_transform = transforms.Compose([
    transforms.Grayscale(),
    transforms.ToTensor(),
])
"""
start_time = time.time()
print("loading train_loader")
#train_loader_input = img_to_tensorVariable(dir_input, namelist_input, 1, 10000)
simple_dataset = folder2.ImageFolder(
    root = '/hoem04/outofhome/MURA_TRAIN_RESIZE',
    #transform = data_transform,
    loader = folder2.pil_loader
)
Loader = torch.utils.data.DataLoader(
    simple_dataset, batch_size = 16, shuffle = False)

#train_loader_output = img_to_tensorVariable(dir_output, namelist_output, 1, 100)
print("loading complete")
print("--- %s seconds ---" %(time.time() - start_time))

ar = np.array(simple_dataset[0][0])
print(ar.shape)



