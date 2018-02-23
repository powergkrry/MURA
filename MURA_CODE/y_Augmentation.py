from PIL import Image
from torchvision import transforms
import folder2
import torch
import numbers
import numpy as np
import matplotlib.pyplot as plt

class FullCrop(object):
    def __init__(self, size):
        self.size = size
        if isinstance(size, numbers.Number):
            self.size = (int(size), int(size))
        else:
            assert len(size) == 2, "Please provide only two dimensions (h, w) for size."
            self.size = size

    def __call__(self, img):
        return fullcrop(img, self.size)

    def __repr__(self):
        return self.__class__.__name__ + '(size={0})'.format(self.size)


def fullcrop(img, size):
    if isinstance(size, numbers.Number):
        size = (int(size), int(size))
    else:
        assert len(size) == 2, "Please provide only two dimensions (h, w) for size."
    
    w, h = img.size
    crop_h, crop_w = size
    if w % crop_w != 0 or h % crop_h != 0:
        raise ValueError("Requested crop size {} must divide input size {}".format(size,(h, w)))

    result = [img.crop(
                     (j*crop_w, i*crop_h, (j+1)*crop_w, (i+1)*crop_h)
              ) for i in range(h//crop_h) for j in range(w//crop_w)]

    #print(result.shape)

    return result


if __name__ == '__main__':
    dir_clean = '/home/powergkrry/MURA/MURA_TRAIN_RESIZE/'

    trans_comp = transforms.Compose([
        #transforms.CenterCrop(100),
        FullCrop((32, 35)),
        transforms.Lambda(lambda crops: torch.stack([transforms.ToTensor()(crop) for crop in crops]))
    ])

    folder_output = folder2.ImageFolder(
        root = dir_clean,
        transform = trans_comp,
        loader = folder2.pil_loader
    )
    folder_output2 = folder2.ImageFolder(
        root = dir_clean,
        transform = transforms.ToTensor(),
        loader = folder2.pil_loader
    )
    train_loader_clean = torch.utils.data.DataLoader(
        folder_output, batch_size = 1, shuffle = False)
    train_loader_clean2 = torch.utils.data.DataLoader(
        folder_output2, batch_size = 1, shuffle = False)


    print(train_loader_clean)

    it = iter(train_loader_clean)
    img, _ = it.next()

    it2 = iter(train_loader_clean2)
    img2, _ = it2.next()

    print(img)
    
    f, a = plt.subplots(16, 10, figsize=(32,35))
    for i in range(16):
        for j in range(10):
            a[i][j].imshow(img.numpy()[0][i*10+j].reshape(32, 35))
            a[i][j].set_xticks(()) 
            a[i][j].set_yticks(())

    plt.show()

    print(img2.shape)

    plt.imshow(img2.numpy()[0][0])
    plt.show()
