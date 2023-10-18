import torch
import glob
import torch.nn as nn
import torch.nn.functional as F
from DropBlock_attention import DropBlock2D
import math
def conv1x1x1(in_planes,out_planes, stride=1, groups=1):
    """3x3 convolution with padding"""
    return nn.Conv3d(in_planes, out_planes, kernel_size=(1,1,1), stride=stride,
                     padding=(0,0,0), groups=groups, bias=False)

def convg3(in_planes,out_planes, stride=1, groups=1):
    """3x3 convolution with padding"""
    return nn.Conv3d(in_planes, out_planes, kernel_size=(1,1,3), stride=stride,
                     padding=(0,0,1), groups=groups, bias=False)

def convg1(in_planes, out_planes, stride=1, groups=1):
    """3x3 convolution with padding"""
    return nn.Conv3d(in_planes, out_planes, kernel_size=(1,1,1), stride=stride,
                     padding=(0,0,0), groups=groups, bias=False)
    
def convk3(in_planes,out_planes, stride=1, groups=1):
    """3x3 convolution with padding"""
    return nn.Conv3d(in_planes, out_planes, kernel_size=(3,3,1), stride=stride,
                     padding=(1,1,0), groups=groups, bias=False)
def convk1(in_planes,out_planes, stride=1, groups=1):
    """3x3 convolution with padding"""
    return nn.Conv3d(in_planes, out_planes, kernel_size=(1,1,1), stride=stride,
                     padding=(0,0,0), groups=groups, bias=False)
def convnb(in_planes, out_planes, stride=1, groups=1):
    """3x3 convolution with padding"""
    return nn.Conv3d(in_planes, in_planes, kernel_size=(3,3,3), stride=stride,
                     padding=(1,1,1), groups=groups, bias=False)

def conv1x1(in_planes, out_planes, stride=1):
    """1x1 convolution"""
    return nn.Conv3d(in_planes, out_planes, kernel_size=(1,1,1),padding=(0,0,0),stride=stride, bias=False)
def conv3x3(in_planes, out_planes, stride=1, groups=1):
    """3x3 convolution with padding"""
    return nn.Conv3d(in_planes, out_planes, kernel_size=(3,3,1), stride=stride,
                     padding=(1,1,0), groups=groups, bias=False)
def conv3x1(in_planes, out_planes, stride=1, groups=1):
    """3x3 convolution with padding"""
    return nn.Conv3d(in_planes, out_planes, kernel_size=(1,1,1), stride=stride,
                     padding=(0,0,0), groups=groups, bias=False)
def conv1x3(in_planes, out_planes, stride=1, groups=1):
    """3x3 convolution with padding"""
    return nn.Conv3d(in_planes, out_planes, kernel_size=(1,1,3), stride=stride,
                     padding=(0,0,1), groups=groups, bias=False)
def conv1x5(in_planes, out_planes, stride=1, groups=1):
    """3x3 convolution with padding"""
    return nn.Conv3d(in_planes, out_planes, kernel_size=(1,1,5), stride=stride,
                     padding=(0,0,2), groups=groups, bias=False)
def conv1x7(in_planes, out_planes, stride=1, groups=1):
    """3x3 convolution with padding"""
    return nn.Conv3d(in_planes, out_planes, kernel_size=(1,1,7), stride=stride,
                     padding=(0,0,3), groups=groups, bias=False)
def conv1x9(in_planes, out_planes, stride=1, groups=1):
    """3x3 convolution with padding"""
    return nn.Conv3d(in_planes, out_planes, kernel_size=(1,1,9), stride=stride,
                     padding=(0,0,4), groups=groups, bias=False)
                     
class M_ResBlock1(nn.Module):
    """M_ResBlock module"""
    def __init__(self, band, inter_size):
        super(M_ResBlock1, self).__init__()

        self.conv1 = nn.Conv3d(in_channels=1, out_channels=24,
                               kernel_size=(1, 1, 7), stride=(1, 1, 2))
        # Dense block
        self.batch_norm1 = nn.Sequential(
            nn.BatchNorm3d(24, eps=0.001, momentum=0.1, affine=True),  # 动量默认值为0.1
            nn.LeakyReLU()
        )
        self.conv2 = nn.Conv3d(in_channels=48, out_channels=12, padding=(0, 0, 3),
                               kernel_size=(1, 1, 7), stride=(1, 1, 1))
        self.batch_norm2 = nn.Sequential(
            nn.BatchNorm3d(60, eps=0.001, momentum=0.1, affine=True),
            nn.LeakyReLU()
        )
        self.conv3 = nn.Conv3d(in_channels=60, out_channels=12, padding=(0, 0, 3),
                               kernel_size=(1, 1, 7), stride=(1, 1, 1))
        self.batch_norm3 = nn.Sequential(
            nn.BatchNorm3d(72, eps=0.001, momentum=0.1, affine=True),
            nn.LeakyReLU()
        )
        self.conv4 = nn.Conv3d(in_channels=72, out_channels=12, padding=(0, 0, 3),
                               kernel_size=(1, 1, 7), stride=(1, 1, 1))
        self.batch_norm4 = nn.Sequential(
            nn.BatchNorm3d(84, eps=0.001, momentum=0.1, affine=True),
            nn.LeakyReLU(inplace=True)
        )
        kernel_3d = math.ceil((band - 6) / 2)
        # print(kernel_3d)
        self.conv5 = nn.Conv3d(in_channels=84, out_channels=band, padding=(0, 0, 0),
                               kernel_size=(1, 1, kernel_3d), stride=(1, 1, 1))

        self.batch_norm5 = nn.Sequential(
            nn.BatchNorm3d(1, eps=0.001, momentum=0.1, affine=True),
            nn.LeakyReLU()
        )

    def forward(self, x):
        x1 = self.conv1(x)
        # print('x11', x11.shape)
        x2 = self.batch_norm1(x1)

        x11 = self.conv1(x)
        # print('x11', x11.shape)
        x21 = self.batch_norm1(x11)
        x21 = torch.cat((x11, x21), dim=1)

        x2 = self.conv2(x21)
        # print('x12', x12.shape)
        x3 = torch.cat((x21, x2), dim=1)
        # print('x13', x13.shape)
        x3 = self.batch_norm2(x3)
        x3 = self.conv3(x3)
        # print('x13', x13.shape)

        x4 = torch.cat((x21, x2, x3), dim=1)
        x4 = self.batch_norm3(x4)
        x4 = self.conv4(x4)

        x5 = torch.cat((x21, x2, x3, x4), dim=1)
        # print('x15', x15.shape)

        # print(x5.shape)
        x6 = self.batch_norm4(x5)
        out = self.conv5(x6)

        #out = self.conv1(a3)  # (128,49,9,9)
        #out = self.bn_in( out)
       # out = F.relu(out)

        #print('1:',out.shape)
        return out




class CDC_MDAA(nn.Module):
    def __init__(self,  band, classes,img_rows,img_cols):
        super(CDC_MDAA, self).__init__()
        self.name = 'CDC_MDAA'
        if classes==16:    
            msize=16
            inter_size=49

        if classes==13: 
            msize=13 
            inter_size=85
        if classes==9:
            msize=9 
            inter_size=49
        # print( 'INPUT：',inter_size)
#         self.layer1 = SPCModuleIN_(1, 1, inter_size=inter_size)
#         self.bn1 = nn.BatchNorm2d(inter_size)
        self.resblock1= M_ResBlock1(band, inter_size)
        self.layer1 = SARes(inter_size, ratio=8,img_rows=img_rows,img_cols=img_cols)
        self.resblock2 = M_ResBlock2(1, inter_size,band)
        self.layer3_1 = SPC3(inter_size, outplane=inter_size, kernel_size=[inter_size, 1, 1], padding=[0, 0, 0],img_rows=img_rows,img_cols=img_cols)
        self.layer3_2 = SPC3(inter_size, outplane=inter_size, kernel_size=[inter_size, 1, 1], padding=[0, 0, 0],img_rows=img_rows,img_cols=img_cols)
        self.layer1_1 = nn.Conv2d(inter_size, inter_size, 1)
        self.layer1_2 = nn.Conv2d(inter_size, inter_size, kernel_size=3, padding=1)
        self.layer1_3 = nn.Conv2d(inter_size, inter_size, 1)
        self.drop = DropBlock2D()
        '''
        self.layer1 = nn.Conv2d(band, inter_size, 1)
       # self.dropblock1 = DropBlock2D(0.6, 5)
        self.dropblock1 = nn.BatchNorm2d(inter_size)
        self.layer1_1 = nn.Conv2d(inter_size, inter_size, kernel_size=3, padding=1)
        #self.dropblock2 = DropBlock2D(0.6, 5)
        self.dropblock2 = nn.BatchNorm2d(inter_size)
        self.layer1_2 = nn.Conv2d(inter_size, inter_size, kernel_size=5, padding=2)
        #self.dropblock3 = DropBlock2D(0.6, 5)
        self.dropblock3 = nn.BatchNorm2d(inter_size)

        self.layer2 = SARes(inter_size, ratio=8)

        self.layer3 = nn.Conv3d(1, inter_size, kernel_size=(band, 1, 1), padding=(0, 0, 0))          
        self.bn3_1 = nn.BatchNorm3d(inter_size)
        self.layer32 = nn.Conv3d(inter_size, inter_size, kernel_size=(3, 3, 3), padding=(1, 1, 1)) 
        self.bn32_1 = nn.BatchNorm3d(inter_size) 
        self.layer33 = nn.Conv3d(inter_size, inter_size, kernel_size=(5, 5, 5), padding=(2, 2, 2)) 
        self.bn33_1 = nn.BatchNorm3d(inter_size)  

        self.layer3_1 = SPC3(inter_size, outplane=inter_size, kernel_size=[inter_size, 1, 1], padding=[0, 0, 0])
        self.layer3_2 = SPC3(inter_size, outplane=inter_size, kernel_size=[inter_size, 1, 1], padding=[0, 0, 0])
        self.layer3_3 = SPC3(inter_size, outplane=inter_size, kernel_size=[inter_size, 1, 1], padding=[0, 0, 0])
'''
        self.layer3 = nn.Conv2d(band, inter_size, 1)
        self.layer4 = nn.Conv2d(inter_size, classes, kernel_size=1)
        self.bn4 = nn.BatchNorm2d(classes)
        self.fc1 = nn.Linear(msize, classes)
        self.fc2 = nn.Linear(msize, classes)
        self.bn5 = nn.BatchNorm2d(inter_size)
    def forward(self, x):
        n, _, h, w, c = x.size()
        #print('x:',x.shape)
        x1 = self.resblock1(x)

        #x1 = x1.squeeze(1).permute(0, 3, 1, 2)#in(128,200,9,9) ksc(176),up(103)，sv(204)
        x1 = x1.permute(0, 4, 1, 2, 3).squeeze(1)  # (128,200,9,9)
        # print('out1',  x1.shape)
        x1 = self.drop( x1)
        # print('out2',  x1.shape)
        x1 = self.layer3(x1)
        x1 = self.bn5(x1)
        x1 = self.layer1(x1)
        #print('x1',x1.shape)
        #x1 =self.layer1_1(x11)
        #x1 =self.layer1_2(x1)
       # x1 =self.layer1_3(x1)
      #  x1 = F.relu(x1+ x11)
        #x1 = self.bn5(x1)
        #print('out2', x1.shape)
        x =self.resblock2(x)
        #print('out4', x.shape)
        #print('out3', x.shape)
        x2 = self.layer3_1(x)
        x2 = F.leaky_relu(x2)
        #print('x2', x2.shape)
        x2 = self.layer3_2(x2)
        x2 = self.drop(x2)
       # x2 = self.layer3_3(x2)

        x = F.leaky_relu(x1+x2)
        #x = torch.cat((x1, x2), dim=1)
        #print('xa:',x.shape)
        x = self.bn4(F.leaky_relu(self.layer4(x)))  # 输入变五倍，输出不变
        # print('注意力后得',x1.shape)
        x = F.avg_pool2d(x, x.size()[-1])
        #x = torch.nn.functional.dropout(x, p=0.5,training=self.training)
        x = self.fc1(x.squeeze())
        #x = self.fc2(x.squeeze())
        return x
'''
        x1 = self.layer1(x.squeeze(1).permute(0, 3, 1, 2))#(16,49,9,9)
        x1 = self.dropblock1(x1)
        x11 = self.layer1_1(x1)
        x1 = self.dropblock2(x11)
        x1 = F.relu(x1)
        x1 = self.layer1_2(x1)
        x1 = self.dropblock3(x1)
        x1 = F.relu(x1+x11)
        x1 = self.layer2(x1)
      
        x = self.layer3(x.permute(0, 1, 4, 2, 3))
        x2 = self.bn3_1(x)
        x = self.layer32(x2)
        x = self.bn32_1(x)
        x = F.relu(x)
        x = self.layer33(x)
        x = self.bn33_1(x).squeeze(2)
        x = F.relu(x2 + x)
        x2 = self.layer3_1(x)
        x2 = self.layer3_2(x2)   
       # x2 = self.layer3_3(x2)
        x2 = F.relu(x2 + x)
       
        x = torch.cat((x1,x2),dim=1)
        x = self.bn4(F.leaky_relu(self.layer4(x)))  # 输入变五倍，输出不变
        #print('注意力后得',x1.shape)
        x = F.avg_pool2d(x, x.size()[-1])
        x = self.fc(x.squeeze())
    # print('x:',x.shape)
'''


'''
class M_ResBlock1(nn.Module):
    """M_ResBlock module"""
    def __init__(self, band, inter_size):
        super(M_ResBlock1, self).__init__()

        self.conv1_in = nn.Conv2d(in_channels=band, out_channels=inter_size, kernel_size=1,)
        self.conv1 = nn.Conv2d(in_channels=inter_size, out_channels=inter_size, kernel_size=1)
        self.conv1_out = nn.Conv2d(in_channels=inter_size, out_channels=band, kernel_size=1)

        self.conv3 = nn.Conv2d(in_channels=inter_size, out_channels=inter_size, kernel_size=3,padding=1)
        self.conv5 = nn.Conv2d(in_channels=inter_size, out_channels=inter_size, kernel_size=5,padding=2)

        self.bn_in = nn.BatchNorm2d(inter_size)
        self.bn_out = nn.BatchNorm2d(band)
    def forward(self, x):
       # a = self.conv1_in(x)
        #a = self.bn_in(a)
       # a = self.conv1(a)
       # a = self.bn_in(a)
       # a = self.conv1_out(a)
        #a = self.bn_out(a)
        #out = F.relu(x+a)
        out = self.conv1_in(x)  # (128,49,9,9)
        out = self.bn_in( out)
        out = F.leaky_relu(out)
        #print(out.shape)
        return out
        
        b = self.conv1_in(block1)
        b = self.bn_in(b)
        b = self.conv3(b)
        b = self.bn_in(b)
        b = self.conv1_out(b)
        b = self.bn_out(b)
        block2= F.relu(block1+b)

        c = self.conv1_in(block2)
        c = self.bn_in(c)
        c = self.conv5(c)
        c = self.bn_in(c)
        c = self.conv1_out(c)
        c = self.bn_out(c)
        out = F.relu(block2 + c)
        
        
'''

class M_ResBlock2(nn.Module):
    """M_ResBlock module"""
    def __init__(self, inchanel, inter_size,band):
        super(M_ResBlock2, self).__init__()
        self.conv1d_in = nn.Conv3d(in_channels=inchanel, out_channels=inter_size, kernel_size=(1, 1, band), padding=(0, 0, 0))
        self.conv1d = nn.Conv3d(in_channels=inter_size, out_channels=inter_size, kernel_size=(1, 1, 1), padding=(0, 0, 0))
        self.conv1d_out = nn.Conv3d(in_channels=inter_size, out_channels=inter_size, kernel_size=(1, 1, 1), padding=(0, 0, 0))
        self.conv3d = nn.Conv3d(in_channels=inter_size, out_channels=inter_size, kernel_size=(3, 3, 3), padding=(1, 1, 1))
        self.conv5d  = nn.Conv3d(in_channels=inter_size, out_channels=inter_size, kernel_size=(5, 5, 5), padding=(2, 2, 2))

        self.bn_in = nn.BatchNorm3d(inter_size)

    def forward(self, x):
        a = self.conv1d_in(x)#(128,49,9,9,1)
        a = self.bn_in(a)
        a = F.leaky_relu(a)
        #print(a.shape)
       # a = self.bn_in(a)
      #  a = self.conv1d(a)
       # a = self.bn_in(a)
        #a = self.conv1d_out(a)
       # x= self.bn_in(a)
       # out = F.relu(x+a)
        #out = self.conv1d_in(out)  # (128,49,9,9,1)
        out =a.squeeze(4)
        #print(out.shape)
        return out


class SpatAttn_(nn.Module):
    """ Position attention module"""

    # Ref from SAGAN
    def __init__(self, in_dim, ratio,img_rows,img_cols):
        super(SpatAttn_, self).__init__()
        self.chanel_in = in_dim
        # 一头
        self.query_conv = nn.Conv2d(in_channels=in_dim, out_channels=in_dim // ratio, kernel_size=1)
        self.key_conv = nn.Conv2d(in_channels=in_dim, out_channels=in_dim // ratio, kernel_size=1)
        self.key1_conv = nn.Conv2d(in_channels=in_dim, out_channels=in_dim // ratio, kernel_size=1)
        self.key2_conv = nn.Conv2d(in_channels=in_dim, out_channels=in_dim // ratio, kernel_size=1)
        self.value_conv = nn.Conv2d(in_channels=in_dim, out_channels=in_dim, kernel_size=1)
        self.bn1 = nn.BatchNorm1d(img_rows*img_cols)
        self.gamma = nn.Parameter(torch.zeros(1))

        self.softmax2 = nn.Softmax(dim=-1)
        self.softmax1 = nn.Softmax(dim=1)
        self.bn = nn.Sequential(
                                nn.BatchNorm2d(in_dim))

        self.query_conv2 = nn.Conv2d(in_channels=in_dim, out_channels=in_dim // ratio, kernel_size=3, padding=1)
        self.key_conv2 = nn.Conv2d(in_channels=in_dim, out_channels=in_dim // ratio, kernel_size=3, padding=1)
        self.key1_conv2 = nn.Conv2d(in_channels=in_dim, out_channels=in_dim // ratio, kernel_size=3, padding=1)
        self.key2_conv2 = nn.Conv2d(in_channels=in_dim, out_channels=in_dim // ratio, kernel_size=3, padding=1)
        self.value_conv2 = nn.Conv2d(in_channels=in_dim, out_channels=in_dim, kernel_size=3, padding=1)

        self.query_conv3 = nn.Conv2d(in_channels=in_dim, out_channels=in_dim // ratio, kernel_size=5, padding=2)
        self.key_conv3 = nn.Conv2d(in_channels=in_dim, out_channels=in_dim // ratio, kernel_size=5, padding=2)
        self.value_conv3 = nn.Conv2d(in_channels=in_dim, out_channels=in_dim, kernel_size=5, padding=2)
        # 四头
        self.query_conv4 = nn.Conv2d(in_channels=in_dim, out_channels=in_dim // ratio, kernel_size=7, padding=3)
        self.key_conv4 = nn.Conv2d(in_channels=in_dim, out_channels=in_dim // ratio, kernel_size=7, padding=3)
        self.value_conv4 = nn.Conv2d(in_channels=in_dim, out_channels=in_dim, kernel_size=7, padding=3)
        self.out_conv = nn.Conv2d(in_channels=98, out_channels=in_dim, kernel_size=1, padding=0)

    def forward(self, x):
        """
            inputs :
                x : input feature maps( B X C X H X W)
            returns :
                out : attention value + input feature
                attention: B X (HxW) X (HxW)
        """
        m_batchsize, C, height, width = x.size()  # BxCxHxW
        proj_query = self.query_conv(x).view(m_batchsize, -1, width * height).permute(0, 2, 1)  # BxHWxC

        proj_key = self.key_conv(x).view(m_batchsize, -1, width * height)  # BxCxHW

        proj_key1 = self.key1_conv(x).view(m_batchsize, -1, width * height).permute(0, 2, 1) # BxCxHW
        proj_key2 = self.key2_conv(x).view(m_batchsize, -1, width * height)  # BxCxHW
        energy0 = torch.bmm(proj_query, proj_key)
        energy1 = torch.bmm(proj_key1, proj_key2)  # BxHWxHW, attention maps
        energy2 = torch.bmm(energy0, energy1)  # BxHWxHW, attention maps

        #print('energy2:',energy2.shape)
        #energy2 = torch.div(energy2,120)
        #energy = self.softmax2(energy2)
        #attention = energy / (1e-9 + energy.sum(dim=1, keepdim=True))
        #attention = self.softmax1(energy2//120)
        attention = self.bn1(energy2//120)  # BxHWxHW, normalized attn maps
        proj_value = self.value_conv(x).view(m_batchsize, -1, width * height)  # BxCxHW

        out = torch.bmm(proj_value, attention.permute(0, 2, 1))  # BxCxHW

        out = out.view(m_batchsize, C, height, width)  # BxCxHxW
        out = self.gamma * out  # + x
        

        proj_query2 = self.query_conv2(x).view(m_batchsize, -1, width * height).permute(0, 2, 1)  # BxHWxC

        proj_key2 = self.key_conv2(x).view(m_batchsize, -1, width * height)  # BxCxHW
        proj1_key2 = self.key1_conv2(x).view(m_batchsize, -1, width * height).permute(0, 2, 1)  # BxCxHW
        proj2_key2 = self.key2_conv2(x).view(m_batchsize, -1, width * height)  # BxCxHW
        energy0 = torch.bmm(proj_query2, proj_key2)
        energy1 = torch.bmm(proj1_key2, proj2_key2)  # BxHWxHW, attention maps


        energy2 = torch.bmm(energy0 , energy1)  # BxHWxHW, attention maps

        #energy2 = torch.div(energy2, 120)
        #energy = self.softmax2(energy2)
        #attention = energy / (1e-9 + energy.sum(dim=1, keepdim=True))
        #attention = self.softmax1(energy2//120)
        attention = self.bn1(energy2//120)  # BxHWxHW, normalized attn maps
        proj_value2 = self.value_conv2(x).view(m_batchsize, -1, width * height)  # BxCxHW

        out2 = torch.bmm(proj_value2, attention.permute(0, 2, 1))  # BxCxHW

        out2 = out2.view(m_batchsize, C, height, width)  # BxCxHxW
        out2 = self.gamma * out2  # + x
        # out = out2+out+out3
        out = torch.mul(out2, out)
        out=F.leaky_relu(out)
        #out = torch.mul(out3, out)
        #out=F.leaky_relu(out)
        # out = torch.mul(out4, out)
        #out = torch.cat((out,out2),dim=1)
        #out = self.out_conv(out)
        return self.bn(out)


class SARes(nn.Module):
    def __init__(self, in_dim, ratio, img_rows,img_cols,resin=False):
        super(SARes, self).__init__()

        # if resin:
        # self.sa1 = SpatAttn(in_dim, ratio)
        # self.sa2 = SpatAttn(in_dim, ratio)
        #  else:
        #self.conv1 = nn.Conv2d(in_channels=98, out_channels=49, kernel_size=1)
        #self.bn = nn.BatchNorm2d(98)
        self.sa1 = SpatAttn_(in_dim, ratio,img_rows,img_cols)
        self.sa2 = SpatAttn_(in_dim, ratio,img_rows,img_cols)
        #self.sa3 = SpatAttn_(in_dim, ratio)

    def forward(self, x):
        identity = x
        x = self.sa1(x)
        x = self.sa2(x)

        #x = self.conv1(x)
        return F.leaky_relu(x+identity)


class SPC3(nn.Module):
    def __init__(self, msize, outplane, kernel_size, padding,img_rows,img_cols,bias=True):
        super(SPC3, self).__init__()
       
        self.convm0 = nn.Conv3d(1, msize, kernel_size=kernel_size, padding=padding)  # generate mask0
        self.convm1 = nn.Conv3d(1, msize, kernel_size=(msize, 3, 3), padding=(0, 1, 1))  # generate mask1
        self.convm2 = nn.Conv3d(1, msize, kernel_size=(msize, 5, 5), padding=(0, 2, 2))  # generate mask1
        self.convm3 = nn.Conv3d(1, msize, kernel_size=(msize, 7, 7), padding=(0, 3, 3))
        self.conv1 = nn.Conv2d(in_channels=147, out_channels=49, kernel_size=1)
        self.bn1 = nn.BatchNorm1d(img_rows*img_cols)
        self.softmax1 = nn.Softmax(dim=1)
        self.softmax2 = nn.Softmax(dim=-1)
        self.bn2 = nn.BatchNorm2d(outplane)
        self.gamma = nn.Parameter(torch.zeros(1))

    def forward(self, x):
        identity = x  # NCHW
        n, c, h, w = identity.size()

        # NCHW ==> NDHW
        k = self.convm0(x.unsqueeze(1)).squeeze(2).view(n, -1, h * w).permute(0, 2, 1)  # BxHWxC
        q = self.convm0(x.unsqueeze(1)).squeeze(2).view(n, -1, w * h)  # BxCxHW
        q1 = self.convm0(x.unsqueeze(1)).squeeze(2).view(n, -1, w * h).permute(0, 2, 1)  # BxCxHW
        q12 = self.convm0(x.unsqueeze(1)).squeeze(2).view(n, -1, w * h)  # BxCxHW
        energy0 = torch.bmm(k, q)
        energy1 = torch.bmm(q1, q12)
        energy = torch.bmm(energy0, energy1)  # BxHWxHW, attention maps
        #print('x:',energy.shape)
        #energy = torch.div(energy, 120)
        #energy = self.softmax2(energy)
        #attention2 = self.softmax1(energy//120)
        #attention2  = energy / (1e-9 + energy.sum(dim=1, keepdim=True))
        attention2 = self.bn1(energy//120)  # BxHWxHW, normalized attn maps
        v = self.convm0(x.unsqueeze(1)).squeeze(2).view(n, -1, w * h)  # BxCxHW
        out = torch.bmm(v, attention2.permute(0, 2, 1))  # BxCxHW
        out = out.view(n, c, h, w)  # BxCxHxW
        out = self.gamma * out  # + x
        

        k2 = self.convm1(x.unsqueeze(1)).squeeze(2).view(n, -1, h * w).permute(0, 2, 1)  # BxHWxC

        q2 = self.convm1(x.unsqueeze(1)).squeeze(2).view(n, -1, w * h)  # BxCxHW
        q21 = self.convm1(x.unsqueeze(1)).squeeze(2).view(n, -1, w * h).permute(0, 2, 1)  # BxCxHW
        q22 = self.convm1(x.unsqueeze(1)).squeeze(2).view(n, -1, w * h)  # BxCxHW

        energy0 = torch.bmm(k2, q2)  # BxHWxHW, attention maps
        energy1 = torch.bmm(q21, q22)
        energy = torch.bmm(energy0, energy1)
        #energy = torch.div(energy, 120)
        #energy = self.softmax2(energy)
        #attention= self.softmax1(energy//120)
        #attention = energy / (1e-9 + energy.sum(dim=1, keepdim=True))
        attention = self.bn1(energy//120)  # BxHWxHW, normalized attn maps
        v2 = self.convm1(x.unsqueeze(1)).squeeze(2).view(n, -1, w * h)  # BxCxHW
        out2 = torch.bmm(v2, attention.permute(0, 2, 1))  # BxCxHW
        out2 = out2.view(n, c, h, w)  # BxCxHxW
        out2 = self.gamma * out2  # + x

        k3 = self.convm2(x.unsqueeze(1)).squeeze(2).view(n, -1, h * w).permute(0, 2, 1)  # BxHWxC
        q3 = self.convm2(x.unsqueeze(1)).squeeze(2).view(n, -1, w * h)  # BxCxHW
        q31 = self.convm2(x.unsqueeze(1)).squeeze(2).view(n, -1, w * h).permute(0, 2, 1)  # BxCxHW
        q32 = self.convm2(x.unsqueeze(1)).squeeze(2).view(n, -1, w * h)  # BxCxHW
        energy0 = torch.bmm(k3, q3)  # BxHWxHW, attention maps
        energy1 = torch.bmm(q31, q32)
        energy = torch.bmm(energy0 , energy1 )
        #energy = torch.div(energy, 120)
        #energy = self.softmax2(energy)
        #attention= self.softmax1(energy//120)
        #attention = energy / (1e-9 + energy.sum(dim=1, keepdim=True))
        attention = self.bn1(energy//120)  # BxHWxHW, normalized attn maps
        v3 = self.convm2(x.unsqueeze(1)).squeeze(2).view(n, -1, w * h)  # BxCxHW
        # 得到v的值
        out3 = torch.bmm(v3, attention.permute(0, 2, 1))  # BxCxHW
        # v与上面得到矩阵相乘
        out3 = out3.view(n, c, h, w)  # BxCxHxW
        out3 = self.gamma * out3  # + x


        out = torch.mul(out, out2)
        out = self.bn2(out)
        out = torch.mul(out, out3)
        out = self.bn2(out)
        #out = torch.cat((out,out2,out3),dim=1)
        #out = self.conv1(out)
        #out = torch.mul(out, out4)
        #out = self.bn2(out)

        return out

