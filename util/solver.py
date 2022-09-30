import torch
import torch.nn as nn
from torchvision.utils import save_image
from model.spadepix2pix import SPADEGenerator,MultiscaleDiscriminator,Encoder
from model.loss import GANLoss, FeatLoss, VGGLoss,KLDLoss
from util.logger import Logger
from torch.utils.data import DataLoader
from munch import Munch
from tqdm import tqdm
import numpy as np
import random
import os,json,shutil

class Solver():
    def __init__(self,args):
        super().__init__()
        self.args = args
        if self.args.color_file:
            assert self.args.label_nc==1
            with open(args.color_file,'r') as f:
                self.colors = json.load(f)
        else:
            self.colors=None

        if self.args.dataset_mode == 'PairDataset':
            from data.pairdataset import PairDataset
            self.dataset =  {
            x : PairDataset(root=args.data_dir,colors=self.colors,split=x,img_size=args.img_size,suffix=args.suffix,random_flip=args.random_flip) for x in ['train','test']
        }
        elif self.args.dataset_mode == 'LabelDataset':
            from data.labeldataset import LabelDataset
            self.dataset =  {
            x : LabelDataset(root=args.data_dir,colors=self.colors,split=x,img_size=args.img_size,suffix=args.suffix,random_flip=args.random_flip) for x in ['train','test']
        }
        else :
            raise (Exception('Wrong Dataset Mode'))

        if args.mode=='train':
            self.data = {
                x : DataLoader( self.dataset[x],batch_size=args.batch_size,shuffle=True,num_workers=args.num_workers) for x in ['train','test']
            }

        
        if torch.cuda.is_available():
            self.device = torch.device('cuda')
        else:
            raise(Exception('no gpu'))
        
        self.nets,self.optims,self.schedulers = self.build_nets()
        for net in self.nets.values():
            net.to(self.device)
        
        self.ganloss = GANLoss()
        self.vggloss = VGGLoss()
        self.featloss = FeatLoss()
        self.kldloss = KLDLoss()

        self.args.checkpoint_dir = os.path.join(self.args.checkpoint_dir,self.args.name)
        self.args.log_dir = os.path.join(self.args.log_dir,self.args.name)
        self.result_dir = os.path.join(self.args.result_dir,self.args.name,self.args.mode)

        if not os.path.exists(self.result_dir):
            os.makedirs(self.result_dir)
        
        if not self.args.resume_dir:
            self.args.resume_dir = self.args.checkpoint_dir
        
        if self.args.restart_train and os.path.exists(self.args.log_dir):
            shutil.rmtree(self.args.log_dir)
        
        if self.args.continue_train:
            self.load_model(latest=True)
            with open(os.path.join(self.args.checkpoint_dir,'iter.txt'),'r') as f:
                epoch,iter = f.read().split(',')
            self.args.resume_epoch = int(epoch)
            self.args.resume_iter = int(iter)
        else:
            if self.args.resume_epoch > 0:
                self.load_model(self.args.resume_epoch)
            
        
        self.logger = Logger(args)

        #global settings
        torch.backends.cudnn.benchmark = True
        seed =42
        torch.manual_seed(seed)
        np.random.seed(seed)
        random.seed(seed)

    
    def build_nets(self):
        nets = Munch(
            generator = SPADEGenerator(self.args.style_dim,self.args.img_size,use_adain=self.args.use_adain,label_nc=self.args.label_nc),
            encoder = Encoder(self.args.style_dim,self.args.img_size),
            discriminator = MultiscaleDiscriminator(num_D=self.args.num_D,n_layers_D=self.args.n_layers_D,input_nc=self.args.label_nc+3),
        )
        G_params = list(nets.generator.parameters())+list(nets.encoder.parameters())
        D_params = list(nets.discriminator.parameters())
        optims = Munch(
            generator = torch.optim.Adam([{'params': G_params, 'initial_lr': self.args.lr}], lr=self.args.lr, betas=(0, 0.999)),
            discriminator = torch.optim.Adam([{'params': D_params, 'initial_lr': self.args.lr}], lr=self.args.lr, betas=(0, 0.999)),
        )
        schedulers = Munch()
        for k in optims.keys():
            schedulers[k] = torch.optim.lr_scheduler.MultiStepLR(optims[k],milestones=[self.args.lm,self.args.um],gamma = self.args.gamma,last_epoch=self.args.resume_epoch-1)
        return nets,optims,schedulers

    def load_model(self,epoch=None,latest=False):
        resume_dir = self.args.resume_dir
        if not os.path.exists(resume_dir):
            return print('model is not exists')

        if not latest:
            for k in self.nets.keys():
                path = os.path.join(resume_dir,'%s_%.4d.pth'%(k,epoch))
                self.nets[k] = torch.load(path)
                print('load model : epoch %s'%epoch)
                
        else:
            for k in self.nets.keys():
                path = os.path.join(resume_dir,'%s_latest.pth'%k)
                self.nets[k] = torch.load(path)
                print('load model : latest')
        return 

    def reparameterize(self, mu, logvar):
        std = torch.exp(0.5 * logvar)
        eps = torch.randn_like(std)
        z = eps.mul(std) + mu
        return z
    
    def compute_g_loss(self,img,seg):
        loss_G = {}
        mu , logvar = self.nets.encoder(img)
        z = self.reparameterize( mu, logvar)
        fake_img = self.nets.generator(seg, z)

        pred_fake = self.nets.discriminator(fake_img, seg)
        pred_real = self.nets.discriminator(img, seg)

        loss_G['GANLoss'] = self.ganloss(pred_fake, True,for_discriminator=False)
        loss_G['VGGLoss'] = self.vggloss(fake_img,img)*self.args.lambda_vgg
        loss_G['FEATLoss'] = self.featloss(pred_fake,pred_real)*self.args.lambda_feat
        loss_G['KLDLoss'] = self.kldloss(mu,logvar)*self.args.lambda_kld

        loss_G['G_Loss'] =  loss_G['GANLoss'] +loss_G['VGGLoss'] + loss_G['FEATLoss'] + loss_G['KLDLoss']

        return fake_img,loss_G

    def compute_d_loss(self,img,seg):
        loss_D = {}
        with torch.no_grad():
            mu , logvar = self.nets.encoder(img)
            z = self.reparameterize( mu, logvar)
            fake_img = self.nets.generator(seg, z)
        # Fake Detection and Loss
        pred_fake = self.nets.discriminator(fake_img, seg)
        loss_D['FAKE'] = self.ganloss(pred_fake, False,for_discriminator=True)
        # Real Detection and Loss
        pred_real = self.nets.discriminator(img, seg)
        loss_D['REAL'] = self.ganloss(pred_real, True,for_discriminator=True)
        loss_D['D_Loss'] = loss_D['FAKE']+loss_D['REAL']
        return fake_img, loss_D

    def train(self):
        if self.args.resume_iter == 0:
            iter = self.args.resume_epoch*len(self.data['train'])
        else:
            iter = self.args.resume_iter

        for epoch in tqdm(range(self.args.resume_epoch,self.args.total_epoch)):
            print()
            for i, (img,seg) in enumerate(self.data['train']):
                img = img.to(self.device)
                seg = seg.to(self.device)

                # Backprop
                fake_img_D, loss_D = self.compute_d_loss(img,seg)
                self.optims.discriminator.zero_grad()
                loss_D['D_Loss'].backward()
                self.optims.discriminator.step()

                fake_img_G,loss_G = self.compute_g_loss(img,seg)
                self.optims.generator.zero_grad()
                loss_G['G_Loss'].backward()
                self.optims.generator.step()

                if (iter+1)%(self.args.display_every_iter) == 0:
                    sample_image = self.logger.visualizer(self.dataset['train'],seg,fake_img_G,img)
                    print('epoch:{} iter:{} '.format(epoch+1,iter+1)) 
                    self.logger.save_logs(iter+1,loss_G)
                    self.logger.save_logs(iter+1,loss_D)
                    self.logger.save_logs(iter+1,sample_image,type='img')

                if (iter+1) % self.args.save_every_iter == 0:
                    self.logger.save_model(self.nets)
                    with open(os.path.join(self.args.checkpoint_dir,'iter.txt'),'w') as f:
                        f.write('%s,%s'%(epoch,iter))
                iter += 1

            save_image(sample_image,os.path.join(self.result_dir,'epoch_{}.png'.format(epoch+1)))
            for scheduler in self.schedulers.values():
                scheduler.step()
            
            if (epoch+1) % self.args.save_every_epoch == 0:
                self.logger.save_model(self.nets,epoch+1)
    
    @torch.no_grad()
    def test(self):
        for i, (img,seg) in enumerate(self.data['test']):
            img = img.to(self.device)
            seg = seg.to(self.device)
            mu , logvar = self.nets.encoder(img)
            z = self.reparameterize( mu, logvar)
            gen_img = self.nets.generator(seg, z)

            sample_image = self.logger.visualizer(self.dataset['test'],seg,gen_img,img)
            save_image(sample_image,os.path.join(self.result_dir,'%s.png'%i))

            
            


    
                

    