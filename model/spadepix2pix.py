import torch
import torchvision
import torch.nn as nn
import torch.nn.functional as F
from model.base_network import BaseNetwork
import numpy as np
import torch.nn.utils.spectral_norm as spectral_norm


class SPADE(nn.Module):
    def __init__(self, norm_nc,ks=3,label_nc=3,nhidden = 128):
        super().__init__()

        self.norm = nn.InstanceNorm2d(norm_nc, affine=False)

        # The dimension of the intermediate embedding space. Yes, hardcoded.
        
        pw = ks // 2
        self.mlp_shared = nn.Sequential(
            nn.Conv2d(label_nc, nhidden, kernel_size=ks, padding=pw),
            nn.ReLU()
        )
        self.mlp_gamma = nn.Conv2d(nhidden, norm_nc, kernel_size=ks, padding=pw)
        self.mlp_beta = nn.Conv2d(nhidden, norm_nc, kernel_size=ks, padding=pw)

    def forward(self, x, s):

        segmap = F.interpolate(s, size=x.size()[2:], mode='nearest')
        actv = self.mlp_shared(segmap)
        gamma = self.mlp_gamma(actv)
        beta = self.mlp_beta(actv)

        # apply scale and bias
        out = (1 + gamma)*self.norm(x)  + beta

        return out
        
class SpadeAdaIN(nn.Module):
    def __init__(self, norm_nc,z_dim,ks=3,label_nc=3,nhidden = 128):
        super().__init__()

        # The dimension of the intermediate embedding space. Yes, hardcoded.
        
        pw = ks // 2
        self.mlp_shared = nn.Sequential(
            nn.Conv2d(label_nc, nhidden, kernel_size=ks, padding=pw),
            nn.ReLU()
        )
        self.mlp_gamma = nn.Conv2d(nhidden, norm_nc, kernel_size=ks, padding=pw)
        self.mlp_beta = nn.Conv2d(nhidden, norm_nc, kernel_size=ks, padding=pw)
        self.fc = nn.Linear(z_dim, norm_nc*2)
        # self.norm_layer = nn.BatchNorm2d(norm_nc,affine=True)
        self.norm_layer = nn.InstanceNorm2d(norm_nc,affine=False)
    
    def forward(self, x, seg,sty):

        segmap = F.interpolate(seg, size=x.size()[2:], mode='nearest')
        actv = self.mlp_shared(segmap)
        gamma_seg = self.mlp_gamma(actv)
        beta_seg = self.mlp_beta(actv)
        h = self.fc(sty)
        h = h.view(h.size(0),h.size(1),1,1)
        gamma_sty, beta_sty = torch.chunk(h, chunks=2, dim=1)
        gamma = gamma_seg + gamma_sty
        beta = beta_seg + beta_sty

        # apply scale and bias
        out = (1 + gamma)*self.norm_layer(x)  + beta

        return out
    
class SPADEAdaINResBlk(nn.Module):

    def __init__(self, fin, fout, z_dim,label_nc=3):
        super().__init__()
        # Attributes
        self.learned_shortcut = (fin != fout)
        self.z_dim = z_dim
        fmiddle = min(fin, fout)

        # create conv layers
        self.conv_0 = self.norm_layer(nn.Conv2d(fin, fmiddle, kernel_size=3, padding=1))  
        self.conv_1 = self.norm_layer(nn.Conv2d(fmiddle, fout, kernel_size=3, padding=1))
        if self.learned_shortcut:
            self.conv_s = self.norm_layer(nn.Conv2d(fin, fout, kernel_size=1, bias=False))

        self.norm_0 = SpadeAdaIN(fin,z_dim,label_nc=label_nc)
        self.norm_1 = SpadeAdaIN(fmiddle,z_dim,label_nc=label_nc)
        if self.learned_shortcut:
            self.norm_s = SpadeAdaIN(fin,z_dim,label_nc=label_nc)
    
    def get_out_channel(self,layer):
        if hasattr(layer, 'out_channels'):
            return getattr(layer, 'out_channels')
        return layer.weight.size(0)
    
    def norm_layer(self,layer):
        # norm_layer = nn.BatchNorm2d(self.get_out_channel(layer),affine=True)
        # norm_layer = nn.InstanceNorm2d(self.get_out_channel(layer),affine=False)
        # return nn.Sequential(layer, norm_layer)
        norm_layer = spectral_norm
        return norm_layer(layer)

    # note the resnet block with SPADE also takes in |seg|,
    # the semantic segmentation map as input
    def forward(self, x, seg,sty):
        x_s = x

        dx = self.conv_0(F.leaky_relu(self.norm_0(x, seg,sty),2e-1))
        dx = self.conv_1(F.leaky_relu(self.norm_1(dx, seg,sty),2e-1))

        if self.learned_shortcut:
            x_s = self.conv_s(self.norm_s(x, seg,sty))

        out = x_s + dx

        return out

class SPADEResBlk(nn.Module):
    def __init__(self, fin, fout, z_dim=None,label_nc=3):
        super().__init__()
        # Attributes
        self.learned_shortcut = (fin != fout)
        self.z_dim = z_dim
        fmiddle = min(fin, fout)

        # create conv layers
        self.conv_0 = nn.Sequential(nn.Conv2d(fin, fmiddle, kernel_size=3, padding=1),nn.InstanceNorm2d(fmiddle))
        self.conv_1 = nn.Sequential(nn.Conv2d(fmiddle, fout, kernel_size=3, padding=1),nn.InstanceNorm2d(fout))
        if self.learned_shortcut:
            self.conv_s = nn.Sequential(nn.Conv2d(fin, fout, kernel_size=1, bias=False),nn.InstanceNorm2d(fout))

        self.norm_0 = SPADE(fin,label_nc=label_nc)
        self.norm_1 = SPADE(fmiddle,label_nc=label_nc)
        if self.learned_shortcut:
            self.norm_s = SPADE(fin,label_nc=label_nc)

    # note the resnet block with SPADE also takes in |seg|,
    # the semantic segmentation map as input
    def forward(self, x, seg,sty=None):
        x_s = x
        dx = self.conv_0(F.leaky_relu(self.norm_0(x, seg),2e-1))
        dx = self.conv_1(F.leaky_relu(self.norm_1(dx, seg),2e-1))

        if self.learned_shortcut:
            x_s = self.conv_s(self.norm_s(x, seg))

        out = x_s + dx

        return out



class SPADEGenerator(BaseNetwork):
    def __init__(self,z_dim,img_size,nf=64,num_up_layers=5,use_adain=False,label_nc=3):
        super().__init__()
        self.nf = nf
        self.sw = img_size // (2**num_up_layers)
        self.sh = self.sw
        self.z_dim = z_dim

        self.fc = nn.Linear(z_dim, 16 * nf * self.sw * self.sh)
        self.conv = nn.Conv2d(label_nc, 16 * nf, 3, padding=1)

        if use_adain:
            ResBlk = SPADEAdaINResBlk
        else:
            ResBlk = SPADEResBlk

        self.head_0 = ResBlk(16 * nf, 16 * nf,z_dim=z_dim,label_nc=label_nc)

        self.G_middle_0 = ResBlk(16 * nf, 16 * nf,z_dim=z_dim,label_nc=label_nc)
        self.G_middle_1 = ResBlk(16 * nf, 16 * nf,z_dim=z_dim,label_nc=label_nc)

        self.up_0 = ResBlk(16 * nf, 8 * nf,z_dim=z_dim,label_nc=label_nc)
        self.up_1 = ResBlk(8 * nf, 4 * nf,z_dim=z_dim,label_nc=label_nc)
        self.up_2 = ResBlk(4 * nf, 2 * nf,z_dim=z_dim,label_nc=label_nc)
        self.up_3 = ResBlk(2 * nf, 1 * nf,z_dim=z_dim,label_nc=label_nc)

        final_nc = nf

        self.conv_img = nn.Conv2d(final_nc, 3, 3, padding=1)

        self.up = nn.Upsample(scale_factor=2)

    def forward(self, seg, sty):
        z = F.interpolate(seg, size=(self.sh, self.sw))
        x = self.conv(z)

        x = x.view(-1, 16 * self.nf, self.sh, self.sw)
        x = self.head_0(x, seg,sty)
        x = self.up(x)
        x = self.G_middle_0(x, seg,sty)
        x = self.G_middle_1(x, seg,sty)
        x = self.up(x)
        x = self.up_0(x, seg,sty)
        x = self.up(x)
        x = self.up_1(x, seg,sty)
        x = self.up(x)
        x = self.up_2(x, seg,sty)
        x = self.up(x)
        x = self.up_3(x, seg,sty)
        x = self.conv_img(F.leaky_relu(x, 2e-1))
        x = torch.sigmoid(x)
        return x

class Encoder(BaseNetwork):

    def __init__(self,z_dim,img_size,nf=64):
        super().__init__()

        kw = 3
        pw = int(np.ceil((kw - 1.0) / 2))
        ndf = nf

        self.z_dim = z_dim
        self.img_size = img_size
        self.layer1 = nn.Sequential(nn.Conv2d(3, ndf, kw, stride=2, padding=pw), nn.InstanceNorm2d(ndf))
        self.layer2 = nn.Sequential(nn.Conv2d(ndf * 1, ndf * 2, kw, stride=2, padding=pw), nn.InstanceNorm2d(ndf*2))
        self.layer3 = nn.Sequential(nn.Conv2d(ndf * 2, ndf * 4, kw, stride=2, padding=pw),nn.InstanceNorm2d(ndf*4))
        self.layer4 = nn.Sequential(nn.Conv2d(ndf * 4, ndf * 8, kw, stride=2, padding=pw),nn.InstanceNorm2d(ndf*8))
        self.layer5 = nn.Sequential(nn.Conv2d(ndf * 8, ndf * 8, kw, stride=2, padding=pw),nn.InstanceNorm2d(ndf*8))
        if img_size >= 256:
            self.layer6 = nn.AdaptiveAvgPool2d(output_size=(4, 4))

        self.fc_mu = nn.Linear(ndf * 8 * 4 * 4, z_dim)
        self.fc_var = nn.Linear(ndf * 8 * 4* 4, z_dim)

        self.actvn = nn.LeakyReLU(0.2, False)

    def forward(self, x):
        
        x = self.layer1(x)
        x = self.layer2(self.actvn(x))
        x = self.layer3(self.actvn(x))
        x = self.layer4(self.actvn(x))
        x = self.layer5(self.actvn(x))
        if self.img_size >= 256:
            x = self.layer6(self.actvn(x))
        x = self.actvn(x)

        x = x.view(x.size(0), -1)
        mu = self.fc_mu(x)
        logvar = self.fc_var(x)
        return mu, logvar

class MultiscaleDiscriminator(BaseNetwork):

    def __init__(self, num_D=2,n_layers_D=3,input_nc=6):
        super().__init__()
        self.num_D = num_D
        self.n_layers_D=n_layers_D
        self.input_nc=input_nc
        for i in range(num_D):
            subnetD = self.create_single_discriminator()
            self.add_module('discriminator_%d' % i, subnetD)

    def create_single_discriminator(self):
        netD = NLayerDiscriminator(n_layers_D=self.n_layers_D,nf=64,input_nc = self.input_nc)
        return netD

    def downsample(self, input):
        return F.avg_pool2d(input, kernel_size=3,
                            stride=2, padding=[1, 1],
                            count_include_pad=False)

    # Returns list of lists of discriminator outputs.
    # The final result is of size opt.num_D x opt.n_layers_D
    def forward(self, img,seg):
        input = torch.cat([img,seg],1)
        result = []
        for  D in self.children():
            out = D(input)
            result.append(out)
            input = self.downsample(input)
        return result
    

class NLayerDiscriminator(BaseNetwork):
    def __init__(self, n_layers_D=3,nf=64,input_nc = 6):
        super().__init__()

        kw = 4
        padw = int(np.ceil((kw - 1.0) / 2))
        

        sequence = [[nn.Conv2d(input_nc, nf, kernel_size=kw, stride=2, padding=padw),
                     nn.LeakyReLU(0.2, False)]]

        for n in range(1, n_layers_D):
            nf_prev = nf
            nf = min(nf * 2, 512)
            stride = 1 if n == n_layers_D - 1 else 2
            sequence += [[nn.Conv2d(nf_prev, nf, kernel_size=kw,stride=stride, padding=padw),
                          nn.InstanceNorm2d(nf),
                          nn.LeakyReLU(0.2, False)
                          ]]

        sequence += [[nn.Conv2d(nf, 1, kernel_size=kw, stride=1, padding=padw)]]

        # We divide the layers into groups to extract intermediate layer outputs
        for n in range(len(sequence)):
            self.add_module('model' + str(n), nn.Sequential(*sequence[n]))

    def forward(self,input):
        results = [input]
        for submodel in self.children():
            intermediate_output = submodel(results[-1])
            results.append(intermediate_output)
        return results[1:]

class VGG19(torch.nn.Module):
    def __init__(self, requires_grad=False):
        super().__init__()
        vgg_pretrained_features = torchvision.models.vgg19(pretrained=True).features
        self.slice1 = torch.nn.Sequential()
        self.slice2 = torch.nn.Sequential()
        self.slice3 = torch.nn.Sequential()
        self.slice4 = torch.nn.Sequential()
        self.slice5 = torch.nn.Sequential()
        for x in range(2):
            self.slice1.add_module(str(x), vgg_pretrained_features[x])
        for x in range(2, 7):
            self.slice2.add_module(str(x), vgg_pretrained_features[x])
        for x in range(7, 12):
            self.slice3.add_module(str(x), vgg_pretrained_features[x])
        for x in range(12, 21):
            self.slice4.add_module(str(x), vgg_pretrained_features[x])
        for x in range(21, 30):
            self.slice5.add_module(str(x), vgg_pretrained_features[x])
        if not requires_grad:
            for param in self.parameters():
                param.requires_grad = False

    def forward(self, X):
        h_relu1 = self.slice1(X)
        h_relu2 = self.slice2(h_relu1)
        h_relu3 = self.slice3(h_relu2)
        h_relu4 = self.slice4(h_relu3)
        h_relu5 = self.slice5(h_relu4)
        out = [h_relu1, h_relu2, h_relu3, h_relu4, h_relu5]
        return out