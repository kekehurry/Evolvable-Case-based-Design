from util.solver import Solver
from util.parser import get_parser
from torchvision.utils import save_image
from torchvision import transforms
from PIL import Image
import numpy as np
import os,random
import torch
from tqdm import tqdm
import matplotlib.pyplot as plt


def load_styles(args):
    style_dir = os.path.join('styles',args.name)
    if os.path.exists(style_dir):
        styles = np.load(os.path.join(style_dir,'styles.npy'))
    else:
        return print('no style file!')
    style = styles[random.choice(range(0,styles.shape[0]))]
    return style

@torch.no_grad()
def generate(solver,seg,name,num=1):
    seg = seg.to(solver.device)
    output_dir = os.path.join(solver.args.result_dir,solver.args.name,'generate_test')
    if not os.path.exists:
        os.makedirs(output_dir)
    for i in range(num):
        style = load_styles(solver.args)
        style = torch.Tensor(style).to(solver.device)
        gen_img = solver.nets.generator(seg,style)
        save_image(gen_img,os.path.join(output_dir,'%s_%s.png'%(name,i)))
    return gen_img


if __name__ == '__main__':
    args = get_parser().parse_args(args=['--name', 'Shenzhen',  '--mode', 'train' ,'--data_dir', 'datasets/Shenzhen',  '--color_file', 'data/Shenzhen.txt', '--img_size', '512', '--batch_size', '1'])
    solver = Solver(args)
    solver.load_model(latest=True)

    for i, (img,seg) in tqdm(enumerate(solver.data['test'])):
        with torch.no_grad():
            if i==0:
                seg1 = seg.cuda()
            img = img.cuda()
            mu , logvar = solver.nets.encoder(img)
            z = solver.reparameterize( mu, logvar)
            fake_img = solver.nets.generator(seg1, z)
            fake_img = transforms.ToPILImage()(fake_img[0])
            plt.imshow(fake_img)
            plt.show()
        if i==10:
            break



