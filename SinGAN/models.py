import torch
import torch.nn as nn
import numpy as np
import torch.nn.functional as F


class ConvBlock(nn.Sequential):
    def __init__(self, in_channel, out_channel, ker_size, padd, stride):
        super(ConvBlock,self).__init__()
        self.add_module('conv',nn.Conv2d(in_channel ,out_channel,kernel_size=ker_size,stride=stride,padding=padd)),
        self.add_module('norm',nn.BatchNorm2d(out_channel)),
        self.add_module('LeakyRelu',nn.LeakyReLU(0.2, inplace=True))

def weights_init(m):
    classname = m.__class__.__name__
    if classname.find('Conv2d') != -1:
        m.weight.data.normal_(0.0, 0.02)
    elif classname.find('Norm') != -1:
        m.weight.data.normal_(1.0, 0.02)
        m.bias.data.fill_(0)
   
class WDiscriminator(nn.Module):
    def __init__(self, opt):
        super(WDiscriminator, self).__init__()
        self.is_cuda = torch.cuda.is_available()
        N = int(opt.nfc)
        self.head = ConvBlock(opt.nc_im,N,opt.ker_size,opt.padd_size,1)
        self.body = nn.Sequential()
        for i in range(opt.num_layer-2):
            N = int(opt.nfc/pow(2,(i+1)))
            block = ConvBlock(max(2*N,opt.min_nfc),max(N,opt.min_nfc),opt.ker_size,opt.padd_size,1)
            self.body.add_module('block%d'%(i+1),block)
        self.tail = nn.Conv2d(max(N,opt.min_nfc),1,kernel_size=opt.ker_size,stride=1,padding=opt.padd_size)

    def forward(self,x):
        x = self.head(x)
        x = self.body(x)
        x = self.tail(x)
        return x


class GeneratorConcatSkip2CleanAdd(nn.Module):
    def __init__(self, opt):
        super(GeneratorConcatSkip2CleanAdd, self).__init__()
        self.is_cuda = torch.cuda.is_available()
        N = opt.nfc
        self.head = ConvBlock(opt.nc_im,N,opt.ker_size,opt.padd_size,1) #GenConvTransBlock(opt.nc_z,N,opt.ker_size,opt.padd_size,opt.stride)
        self.body = nn.Sequential()
        for i in range(opt.num_layer-2):
            N = int(opt.nfc/pow(2,(i+1)))
            block = ConvBlock(max(2*N,opt.min_nfc),max(N,opt.min_nfc),opt.ker_size,opt.padd_size,1)
            self.body.add_module('block%d'%(i+1),block)
        self.tail = nn.Sequential(
            ConvBlock(max(N,opt.min_nfc),max(N,opt.min_nfc),ker_size=opt.ker_size,stride =1,padd=opt.padd_size)
        )
        self.SPADElayer = SPADEResBlk(max(N,opt.min_nfc),opt.nc_im)
        self.last_act = nn.Tanh()

    def forward(self,x,y,seg):
        x = self.head(x)
        x = self.body(x)
        x = self.tail(x)
        #print('before SPADE',x.size())
        x = self.SPADElayer(x, seg)
        x = self.last_act(x)
        
        ind = int((y.shape[2]-x.shape[2])/2)
        y = y[:,:,ind:(y.shape[2]-ind),ind:(y.shape[3]-ind)]
        return x+y

class SPADE(nn.Module):
    def __init__(self, norm_nc, label_nc):#input 채널 갯수, label 채널 갯수
        super().__init__()

        #self.param_free_norm = nn.BatchNorm2d(norm_nc, affine=False) #CJ: 이미 앞에 레이어에서 노말라이즈 하고있음.

        # The dimension of the intermediate embedding space. Yes, hardcoded.
        nhidden = 128

        ks = 3
        self.embedding = nn.Embedding(num_embeddings = 300, embedding_dim=10)
        self.mlp_shared = nn.Sequential(
            nn.Conv2d(label_nc, nhidden, kernel_size=ks, padding=1),
            nn.ReLU()
        )

        self.mlp_gamma = nn.Conv2d(nhidden, norm_nc, kernel_size=ks, padding=1)
        self.mlp_beta = nn.Conv2d(nhidden, norm_nc, kernel_size=ks, padding=1)

    def forward(self, x, segmap): #x:input 채널 갯수, segmentation map
        normalized = x
        segmap = segmap.unsqueeze(0).unsqueeze(0)
        segmap = F.interpolate(segmap.float(), size=x.size()[2:], mode='nearest').long()
        segmap = segmap.squeeze(0).squeeze(0)
        
        embed = self.embedding(segmap)
        embed = embed.permute(2,0,1).unsqueeze(0)
        
        actv = self.mlp_shared(embed)
        
        gamma = self.mlp_gamma(actv)
        beta = self.mlp_beta(actv)

        # apply scale and bias
        out = normalized * (1 + gamma) + beta

        return out

class SPADEResBlk(nn.Module):
    def __init__(self, fin, fout):
        super().__init__()

        fmiddle = min(fin, fout)

        # create conv layers
        self.conv_0 = nn.Conv2d(fin, fmiddle, kernel_size=3, padding=1)
        self.conv_1 = nn.Conv2d(fmiddle, fout, kernel_size=3, padding=1)

        self.norm_0 = SPADE(fin, 10)
        self.norm_1 = SPADE(fmiddle, 10)

    def forward(self, x, seg):
        dx = self.conv_0(self.actvn(self.norm_0(x, seg)))
        dx = self.conv_1(self.actvn(self.norm_1(dx, seg)))

        return dx

    def actvn(self, x):
        return F.leaky_relu(x, 2e-1)