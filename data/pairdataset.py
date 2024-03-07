from configparser import Interpolation
from torch.utils.data import DataLoader, Dataset
from torchvision import transforms
from PIL import Image
import numpy as np
import torch
import random
import cv2
import os
import matplotlib.pyplot as plt

class PairDataset(Dataset):
    def __init__(self,root,colors,split='train',img_size = 256,suffix='.png',random_flip=False):
        self.root = root
        self.split = split
        self.img_size = img_size
        self.files = {}
        self.images_base = os.path.join(self.root,self.split)
        self.files[self.split] = self.recursive_glob(self.images_base)

        self.colors = colors
        if self.colors:
            self.label_colours = dict(zip(range(len(self.colors)),self.colors))

        self.transform = transforms.Compose([
            transforms.Resize(self.img_size,interpolation=transforms.InterpolationMode.NEAREST),
            transforms.ToTensor(),
        ])
    
    def is_image_file(self,filename):
        IMG_EXTENSIONS = ['.jpg', '.JPG', '.jpeg', '.JPEG','.png', '.PNG', '.ppm', '.PPM', '.bmp', '.BMP', '.tiff', '.webp']
        return any(filename.endswith(extension) for extension in IMG_EXTENSIONS)
    
    def recursive_glob(self,rootdir="."):
        return [
            os.path.join(looproot, filename)
            for looproot, _, filenames in os.walk(rootdir)
            for filename in filenames
            if self.is_image_file(filename)
        ]

    def encode_segmap(self,seg):
        if self.colors:
            seg = np.array(seg,dtype=np.uint8)
            for i in range(len(self.colors)):
                color = np.array(self.colors[i])
                mask = cv2.inRange(seg,color-10,color+10)
                mask = cv2.cvtColor(mask,cv2.COLOR_GRAY2RGB)
                seg[mask==255] = i
            seg[seg>=len(self.colors)] = len(self.colors)
            return seg[:,:,0]
        else:
            return seg

    def decode_segmap(self, temp):
        if self.colors:
            r = temp.copy()
            g = temp.copy()
            b = temp.copy()
            for i in range(len(self.colors)):
                r[r == i] = self.colors[i][0]
                g[g == i] = self.colors[i][1]
                b[b == i] = self.colors[i][2]

            rgb = np.zeros((temp.shape[0], temp.shape[1], 3))
            rgb[:, :, 0] = r / 255.0
            rgb[:, :, 1] = g / 255.0
            rgb[:, :, 2] = b / 255.0
            return rgb
        else:
            return temp
        
    
    def __getitem__(self,index):
        img_path = self.files[self.split][index].rstrip()
        raw_img = Image.open(img_path).convert('RGB')
        raw_img = np.array(raw_img)
        h,w,c = raw_img.shape
        seg = raw_img[:,:w//2,:]
        seg = self.encode_segmap(seg)
        seg = self.transform(Image.fromarray(seg))

        img = raw_img[:,w//2:,:]
        img = self.transform(Image.fromarray(img))

        if self.random_flip:
            p1 = random.randint(0,1)
            p2 = random.randint(0,1)
            img = transforms.RandomHorizontalFlip(p1)(img)
            img = transforms.RandomVerticalFlip(p2)(img)
            seg = transforms.RandomHorizontalFlip(p1)(seg)
            seg = transforms.RandomHorizontalFlip(p2)(seg)

        return img, seg
    
    def __len__(self):
        return len(self.files[self.split])


if __name__ == '__main__':

    colors = [[0,0,0],[255,255,255],[0,128,0],[0,0,255],[128,0,128]]
    dataset =  {
            x : PairDataset(root='../datasets/Manhattan',colors=colors,split=x,img_size=256) for x in ['train','test']
        }
    data = {
            x : DataLoader( dataset[x],batch_size=1,shuffle=True,num_workers=0) for x in ['train','test']
        }
    img,seg = next(iter(data['train']))

    print(img.shape)
    print(seg.shape)
    seg = transforms.ToPILImage()(seg[0])
    seg = dataset['train'].decode_segmap(np.array(seg))

    img = transforms.ToPILImage()(img[0])

    ax = plt.subplot(1,2,1)
    ax.imshow(seg)
    
    ax1 = plt.subplot(1,2,2)
    ax1.imshow(img)

    plt.show()