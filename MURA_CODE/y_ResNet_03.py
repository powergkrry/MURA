import torch
import torch.nn as nn
from torch.autograd import Variable
import matplotlib.pyplot as plt
import numpy as np
import os
from PIL import Image
import torchvision
from torchvision import transforms
from torchvision.transforms import ToTensor, ToPILImage
import folder2
import random
import math
import resnet2
from logger import Logger
import y_Augmentation

"""
check :
1. num of epoch
2. start epoch num 
3. load pretrained model, optimizer
"""
"""
This model is DnCNN-like model.
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

     img_arr = Variable(torch.Tensor(img_arr).view(-1, 1, 512, 350)).cuda()
     return img_arr


def cal_test_mse_psnr(test_file_num, BATCH_SIZE):
    test_clean_iter = iter(test_loader_clean)
    loss = 0
    avg_psnr = 0

    for i in range(len(test_clean_iter)):
        data_clean, _ = test_clean_iter.next()
        data_clean = data_clean.type(torch.FloatTensor)

        data_input = add_gaussian_noise(data_clean, 20)
        data_input = data_input.type(torch.FloatTensor)

        x = Variable(data_input.cuda())
        y = Variable(data_clean.cuda())

        test_denoise = autoencoder(x)

        mse = loss_func(test_denoise, y).data[0]
        loss += mse
        psnr = -10 * math.log10(mse)
        avg_psnr += psnr

        if i == 0:
            first_batch = test_denoise

    test_loss_output = loss / int(test_file_num / BATCH_SIZE) # approximate
    test_avg_psnr_output = avg_psnr / int(test_file_num / BATCH_SIZE) # approximate

    return test_loss_output, test_avg_psnr_output, first_batch

def add_gaussian_noise(image_in, noise_sigma):

    temp_image = image_in.clone()

    noise = (torch.randn((*temp_image.shape)) * noise_sigma)/255.

    noisy_image = torch.add(temp_image,noise)

    return noisy_image

trans_comp = transforms.Compose([
        #transforms.CenterCrop(100),
        #transforms.ToTensor()
        y_Augmentation.FullCrop((32, 35)),
        transforms.Lambda(lambda crops: torch.stack([transforms.ToTensor()(crop) for crop in crops]))
    ])

torch.manual_seed(1)    # reproducible
torch.cuda.manual_seed_all(1) # for multi gpu
#torch.backends.cudnn.enabled = False
np.random.seed(1)
random.seed(1)

EPOCH = 30 
BATCH_SIZE = 64
test_BATCH_SIZE = 8
LR1 = 0.001     # learning rate
LR2 = 0.0001
LR3 = 0.00001
LR4 = 0.000005
LR = 0.0001
N_TEST_IMG = 5
train_error = []
test_error = []
train_psnr = []
test_psnr = []

dir_clean = '/home/powergkrry/MURA/MURA_TRAIN_RESIZE/'
dir_test_clean = '/home/powergkrry/MURA/MURA_TEST_RESIZE/'
namelist_clean = os.listdir('/home/powergkrry/MURA/MURA_TRAIN_RESIZE/test/')
namelist_test = os.listdir('/home/powergkrry/MURA/MURA_TEST_RESIZE_NOISE/test/')
train_file_num = len(namelist_clean)
test_file_num = len(namelist_test)

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

print("generating autoencoder")
autoencoder = AutoEncoder()
#autoencoder = resnet2.resnet34()
autoencoder = torch.nn.DataParallel(autoencoder, device_ids = [0, 1])
autoencoder.cuda()

optimizer = torch.optim.Adam(autoencoder.parameters(), lr=LR, weight_decay=1e-5)
#scheduler = torch.optim.lr_scheduler.StepLR(optimizer, step_size=14, gamma=0.1)

"""
# load pretrained weight, optimizer
autoencoder.load_state_dict(torch.load('./pretrained/y_Conv02_epoch10_model.pth.tar'))
optimizer.load_state_dict(torch.load('./pretrained/y_Conv02_epoch10_optimizer.pth.tar'))
"""
loss_func = nn.MSELoss()

"""
print("loading view_data")
view_data_in = img_to_tensorVariable(
    os.path.join(dir_input,'test/'), namelist_input, 1, N_TEST_IMG
)
view_data_out = img_to_tensorVariable(
    os.path.join(dir_output,'test/'), namelist_output, 1, N_TEST_IMG
)
print("loading complete")
"""

print("loading train_loader")
folder_output = folder2.ImageFolder(
    root = dir_clean,
    transform = trans_comp,
    loader = folder2.pil_loader
)
train_loader_clean = torch.utils.data.DataLoader(
    folder_output, batch_size = BATCH_SIZE, shuffle = False)
print("loading complete")

print("loading test_loader")
folder_test_clean = folder2.ImageFolder(
    root = dir_test_clean,
    transform = trans_comp,
    loader = folder2.pil_loader
)
test_loader_clean = torch.utils.data.DataLoader(
    folder_test_clean, batch_size = test_BATCH_SIZE, shuffle = False)
print("loading complete")

print("making noise")
train_iter = iter(train_loader_clean)
train_input = []
for step in range(len(train_iter)):
    data_clean, _ = train_iter.next()
    data_noise = add_gaussian_noise(data_clean, 20)
    train_input.append(data_noise.type(torch.FloatTensor))

print("make noise complete")

"""
# show image
f, a = plt.subplots(3, N_TEST_IMG, figsize=(10,6))
plt.ion()

for i in range(N_TEST_IMG):
    a[0][i].imshow(view_data_in.data.cpu().numpy()[i].reshape(512, 350), cmap='gray')
    a[0][i].set_xticks(()) 
    a[0][i].set_yticks(())
    a[1][i].imshow(view_data_out.data.cpu().numpy()[i].reshape(512,350), cmap='gray')
    a[1][i].set_xticks(())
    a[1][i].set_yticks(())
"""
logger = Logger('./logs')

for epoch in range(EPOCH):
    train_clean_iter = iter(train_loader_clean)

    train_loss_sum = 0
    train_psnr_sum = 0

    if (epoch+1) % 10 == 0:
        LR /= 10
    optimizer = torch.optim.Adam(autoencoder.parameters(), lr=LR, weight_decay=1e-5)
    
    for step in range(len(train_clean_iter)):
        data_clean, _ = train_clean_iter.next()
        data_input = train_input[step]
        data_clean = data_clean.type(torch.FloatTensor)

        x = Variable(data_input.cuda())
        y = Variable(data_clean.cuda())
        decoded = autoencoder(x) # delete encoded,
        
        loss = loss_func(decoded, y)
        optimizer.zero_grad()
        loss.backward()
        optimizer.step()
        
        train_loss_sum += loss.data[0]
        psnr = -10 * math.log10(loss.data[0])
        train_psnr_sum += psnr

#        if True:
        if (step+1) % 100 == 0:
#            break
            print("Epoch :", epoch, "| step :",step,"| train loss: %0.6f" % loss.data[0])
            
            """
            #show images
            final_image = x.data.cpu().numpy() - decoded.data.cpu().numpy()            
            
            f, a = plt.subplots(3, N_TEST_IMG, figsize=(7,7))
            for i in range(N_TEST_IMG):
                a[0][i].imshow(data_input.cpu().numpy()[i].reshape(100, 100), cmap='gray')
                a[0][i].set_xticks(()) 
                a[0][i].set_yticks(())
                a[1][i].imshow(data_clean.cpu().numpy()[i].reshape(100,100), cmap='gray')
                a[1][i].set_xticks(())
                a[1][i].set_yticks(())
           
            for i in range(N_TEST_IMG):
                a[2][i].clear()
                a[2][i].imshow(final_image[i].reshape(100,100), cmap='gray')
                a[2][i].set_xticks(())
                a[2][i].set_yticks(())
            plt.draw()
            #plt.pause(0.5)
            plt.show()
            """

            #TensorBoard logging
         
        if step == 0:  
            """ 
            #(1) Log the scalar values
            logger.scalar_summary('loss', loss.data[0], epoch+1)

            #(2) Log values and gadients of the parameters (histogram)
            for tag, value in autoencoder.named_parameters():
                tag = tag.replace('.', '/')
                logger.histo_summary(tag, value.data.cpu().numpy(), epoch+1)
                logger.histo_summary(tag+'/grad', value.grad.data.cpu().numpy(), epoch+1)
            """
            #(3) Log the images
            info = {
                'train_images': decoded.view(-1, 100, 100).data.cpu().numpy()
            }

            for tag, images in info.items():
                logger.image_summary(tag, images, epoch+1)

    epoch_ = epoch + 0 + 1

# save data
    save_name_model = './pretrained/y_ResNet01_epoch' + str(epoch_) + '_model' + '.pth.tar'
    save_name_optimizer = './pretrained/y_ResNet01_epoch' + str(epoch_) + '_optimizer' + '.pth.tar'
    torch.save(autoencoder.state_dict(), save_name_model)
    torch.save(optimizer.state_dict(), save_name_optimizer)
#    scheduler.step()
    print("model saved")

    this_train_err = train_loss_sum * BATCH_SIZE / train_file_num
    train_error.append(this_train_err) # approximate
    this_train_psnr = train_psnr_sum * BATCH_SIZE / train_file_num
    train_psnr.append(this_train_psnr) # approximate
    test_loss_output, test_avg_psnr_output, first_batch = cal_test_mse_psnr(test_file_num, test_BATCH_SIZE)
    test_error.append(test_loss_output)
    test_psnr.append(test_avg_psnr_output)

    logger.scalar_summary('train_loss', this_train_err, epoch+1)
    logger.scalar_summary('test_loss', test_loss_output, epoch+1)
    logger.scalar_summary('train_psnr', this_train_psnr, epoch+1)
    logger.scalar_summary('test_psnr', test_avg_psnr_output, epoch+1)

    info = {
        'test_images': first_batch.view(-1, 100, 100)[:8].data.cpu().numpy()
    }
    for tag, images in info.items():
        logger.image_summary(tag, images, epoch+1)

    print("\ntrain_error")
    print(train_error)
    print("\ntest_error")
    print(test_error)
    print("\ntrain_psnr =",train_psnr)
    print("\ntest_psnr =", test_psnr)


"""
# Plot Test Image
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

"""
#make result graph
fig = plt.figure(figsize=(12, 6))
ax1 = fig.add_subplot(121)
ax2 = fig.add_subplot(122)
ax1.set_title("train psnr")
ax2.set_title("test psnr")
ax1.plot(train_psnr)
ax2.plot(test_psnr)
plt.savefig("./y_Results/y_ConvUpsample_02_01.png")
"""

