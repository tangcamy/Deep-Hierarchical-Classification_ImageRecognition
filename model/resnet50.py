'''Deep Hierarchical Classifier using resnet50 with cbam as the base.
'''

import torch
import torch.nn as nn
from .cbam import CBAM

class BottleNeck(nn.Module):
    '''Bottleneck modules
    '''

    def __init__(self, in_channels, out_channels, expansion=4, stride=1, use_cbam=True):
        '''Param init.
        '''
        super(BottleNeck, self).__init__()

        self.use_cbam = use_cbam
        #only the first conv will be affected by the given stride parameter. The rest have default stride value (which is 1).
        self.conv1 = nn.Conv2d(in_channels=in_channels, out_channels=out_channels, kernel_size=1, bias=False, stride=stride)
        self.bn1 = nn.BatchNorm2d(num_features=out_channels)
        self.conv2 = nn.Conv2d(in_channels=out_channels, out_channels=out_channels, kernel_size=3, padding=1, bias=False)
        self.bn2 = nn.BatchNorm2d(num_features=out_channels)
        self.conv3 = nn.Conv2d(in_channels=out_channels, out_channels=out_channels*expansion, kernel_size=1, bias=False)
        self.bn3 = nn.BatchNorm2d(num_features=out_channels*expansion)
        self.relu = nn.ReLU(inplace=True)

        #since the input has to be same size with the output during the identity mapping, whenever the stride or the number of output channels are
        #more than 1 and expansion*out_channels respectively, the input, x, has to be downsampled to the same level as well.
        self.identity_connection = nn.Sequential()
        if stride != 1 or in_channels != expansion*out_channels:
            self.identity_connection = nn.Sequential(
                nn.Conv2d(in_channels=in_channels, out_channels=expansion*out_channels, kernel_size=1, stride=stride, bias=False),
                nn.BatchNorm2d(num_features=out_channels*expansion)
            )

        if self.use_cbam:
            self.cbam = CBAM(channel_in=out_channels*expansion)


    def forward(self, x):#x:input_batch
        '''Forward Propagation.
        '''

        out = self.relu(self.bn1(self.conv1(x)))
        out = self.relu(self.bn2(self.conv2(out)))
        out = self.bn3(self.conv3(out))

        if self.use_cbam:
            out = self.cbam(out)

        out += self.identity_connection(x) #identity connection/skip connection
        out = self.relu(out)

        return out


class ResNet50(nn.Module):
    '''ResNet-50 Architecture.
    '''

    def __init__(self, use_cbam=True, image_depth=3, num_classes=[2,4,14]):#三層階層
        '''Params init and build arch.
        '''
        super(ResNet50, self).__init__()

        self.in_channels = 64 
        self.expansion = 4
        self.num_blocks = [3, 4, 6, 3]
        #in_channels:輸入通道數  ; out_channels:輸出通道數(可隨意定義)
        #權重的shape為in_ch * kernelsize(3*3)
        #權重張量(tensor)完整大小:out_ch x in_ch x kernelsize(3x3) = 3*64*3* 3
        #bias:偏值，加在輸出圖片各個通道上面的常數值
        #padding填補邊界：針對圖片的邊緣進行填補，數值為0 : 使其輸出圖片的大小和原始圖片大小一致了
        self.conv_block1 = nn.Sequential(nn.Conv2d(kernel_size=3, stride=1, in_channels=image_depth, out_channels=self.in_channels, padding=1, bias=False),
                                            nn.BatchNorm2d(self.in_channels),
                                            nn.ReLU(inplace=True))
        #stride 步長：提取各軸下個元數時，需要跳過的元素數量
        self.layer1 = self.make_layer(out_channels =64, num_blocks=self.num_blocks[0], stride=1, use_cbam=use_cbam)
        self.layer2 = self.make_layer(out_channels=128, num_blocks=self.num_blocks[1], stride=2, use_cbam=use_cbam)
        self.layer3 = self.make_layer(out_channels=256, num_blocks=self.num_blocks[2], stride=1, use_cbam=use_cbam)
        self.layer4 = self.make_layer(out_channels=512, num_blocks=self.num_blocks[3], stride=2, use_cbam=use_cbam)
        self.avgpool = nn.AdaptiveAvgPool2d((1, 1))

        self.linear_lvl1 = nn.Linear(512*self.expansion, num_classes[0]) #線性flatten輸出
        self.linear_lvl2 = nn.Linear(512*self.expansion, num_classes[1])
        self.linear_lvl3 = nn.Linear(512*self.expansion, num_classes[2]) #add layer3

        self.softmax_reg1 = nn.Linear(num_classes[0], num_classes[0])
        self.softmax_reg2 = nn.Linear(num_classes[0]+num_classes[1], num_classes[1])
        self.softmax_reg3 = nn.Linear(num_classes[0]+num_classes[1]+num_classes[2], num_classes[2]) #add layer3



    def make_layer(self, out_channels, num_blocks, stride, use_cbam):
        '''To construct the bottleneck layers.
        '''
        strides = [stride] + [1]*(num_blocks-1)
        layers = []
        for stride in strides:
            layers.append(BottleNeck(in_channels=self.in_channels, out_channels=out_channels, stride=stride, expansion=self.expansion, use_cbam=use_cbam))
            self.in_channels = out_channels * self.expansion
        return nn.Sequential(*layers)


    def forward(self, x): #__call__ #x:input_batch
        '''Forward propagation of ResNet-50.
        '''

        x = self.conv_block1(x)
        x = self.layer1(x)
        x = self.layer2(x)
        x = self.layer3(x)
        x_conv = self.layer4(x)
        x = self.avgpool(x_conv)
        x = nn.Flatten()(x) #flatten the feature maps.卷積資料攤平

        level_1 = self.softmax_reg1(self.linear_lvl1(x)) # 經過softxmax處理後的模型輸出（機率值）
        level_2 = self.softmax_reg2(torch.cat((level_1, self.linear_lvl2(x)), dim=1))
        level_3 = self.softmax_reg3(torch.cat((level_1,level_2, self.linear_lvl3(x)), dim=1))# add layer3



        return level_1, level_2 ,level_3
