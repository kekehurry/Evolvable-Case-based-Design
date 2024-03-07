import os
import torch
import pandas as pd
import numpy as np
from PIL import Image
from util.solver import Solver
from util.parser import get_parser
from util.calculate import calculate
from torchvision import transforms

args = get_parser().parse_args(args=['--name', 'Shenzhen',  '--mode', 'train' ,'--data_dir', 'datasets/Shenzhen',  '--color_file', 'data/Shenzhen.txt', '--img_size', '512', '--batch_size', '1'])
solver = Solver(args)
solver.load_model(latest=True)
styles = []
df = pd.DataFrame(columns=['ID','FSI','GSI','L','OSR'])

for i, (img,seg) in enumerate(solver.data['train']):
        img = img.to(solver.device)
        seg = seg.to(solver.device)

        # Backprop
        with torch.no_grad():
            mu , logvar = solver.nets.encoder(img)
            z = solver.reparameterize( mu, logvar)
            styles.append(z.cpu().numpy())
            gen_img = solver.nets.generator(seg, z)
            gen_img = transforms.ToPILImage()(gen_img[0])
            gen_img.save('results/Shenzhen/gen/%s.png'%i)
            data,contours,heights,ids = calculate(gen_img)
            df.loc[i] = [i,data[0],data[1],data[2],data[3]]
            print('processing %s/%s...'%(i,len(solver.data['train'])-1))

styles = np.stack(styles,axis=0)
print(styles.shape)
np.save('styles/Shenzhen/styles.npy',styles)
df.to_excel('styles/Shenzhen/generated_data.xlsx')



