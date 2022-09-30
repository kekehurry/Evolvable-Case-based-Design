from torchvision.utils import make_grid
from torch.utils.tensorboard import SummaryWriter
import numpy as np
import torch
import os

class Logger():
    def __init__(self,args):
        self.args = args
        return 

    def save_logs(self,iter,item,type='dict'):
        log_dir = self.args.log_dir
        if not os.path.exists(log_dir):
            os.makedirs(log_dir)
        with SummaryWriter(log_dir=log_dir, comment='train') as writer: 
            if type=='dict':
                for k in item.keys():
                    writer.add_scalar(k, item[k].item(), iter)
                    print('{}:{}'.format(k,item[k].item()),end=' ')
            elif type == 'img':
                writer.add_image('sample_img', item, iter)
            else:
                raise(Exception('Wrong type'))
        return print()
    
    def save_model(self,nets,iter=None):
        checkpoint_dir = self.args.checkpoint_dir
        if not os.path.exists(checkpoint_dir):
            os.makedirs(checkpoint_dir)
        if not iter:
            for k in nets.keys():
                torch.save(nets[k], os.path.join(checkpoint_dir,'%s_latest.pth'%k))
        else:
            for k in nets.keys():
                torch.save(nets[k], os.path.join(checkpoint_dir,'%s_%.4d.pth'%(k,iter)))

    def visualizer(self,dataset,seg,fake,img):
        seg,fake,img = seg.cpu(),fake.cpu(),img.cpu()
        segmap = torch.zeros(seg.size(0), 3, seg.size(2), seg.size(3))
        # Plotting function for segmentation task
        for i, a in enumerate(seg):
            image = a.squeeze()
            image = image.numpy()
            if not self.args.color_file:
                image = image.transpose((1, 2, 0))
                image = dataset.decode_segmap(image*255.)
            else:
                image = dataset.decode_segmap(image*255.)
            image = image.transpose((2, 0, 1))
            image = torch.from_numpy(image)
            segmap[i] = image
        output = torch.cat([segmap,fake,img])
        grid_img = make_grid(output, nrow=seg.size(0))
        return grid_img