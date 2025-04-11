import torch
import torch.nn as nn
import torch.nn.functional as F
from torch.autograd import Variable
from einops import rearrange
from model.videoit import time_T



class Bottleneck(nn.Module):
    expansion = 4

    def __init__(self, inplanes, planes, stride=1, downsample=None, head_conv=1):
        super(Bottleneck, self).__init__()
        if head_conv == 1:
            self.conv1 = nn.Conv3d(inplanes, planes, kernel_size=1, bias=False)
            self.bn1 = nn.BatchNorm3d(planes)
        elif head_conv == 3:
            self.conv1 = nn.Conv3d(inplanes, planes, kernel_size=(3, 1, 1), bias=False, padding=(1, 0, 0))
            self.bn1 = nn.BatchNorm3d(planes)
        else:
            raise ValueError("Unsupported head_conv!")
        self.conv2 = nn.Conv3d(
            planes, planes, kernel_size=(1, 3, 3), stride=(1, stride, stride), padding=(0, 1, 1), bias=False)
        self.bn2 = nn.BatchNorm3d(planes)
        self.conv3 = nn.Conv3d(planes, planes * 4, kernel_size=1, bias=False)
        self.bn3 = nn.BatchNorm3d(planes * 4)
        self.relu = nn.ReLU(inplace=True)
        self.downsample = downsample
        self.stride = stride

    def forward(self, x):
        residual = x

        out = self.conv1(x)
        out = self.bn1(out)
        out = self.relu(out)

        out = self.conv2(out)
        out = self.bn2(out)
        out = self.relu(out)

        out = self.conv3(out)
        out = self.bn3(out)

        if self.downsample is not None:
            residual = self.downsample(x)
        out += residual
        out = self.relu(out)

        return out


class STFusion(nn.Module):
    def __init__(self, block=Bottleneck, layers=[3, 4, 6, 3], num_classes=2, dropout=0.1,):
        super(STFusion, self).__init__()

        self.fast_inplanes = 32
        self.fast_conv1 = nn.Conv3d(3, 32, kernel_size=(5, 7, 7), stride=(1, 2, 2), padding=(2, 3, 3), bias=False)
        self.fast_bn1 = nn.BatchNorm3d(32)
        self.fast_relu = nn.ReLU(inplace=True)
        self.fast_maxpool = nn.MaxPool3d(kernel_size=(1, 3, 3), stride=(1, 2, 2), padding=(0, 1, 1))
        self.fast_res2 = self._make_layer_fast(block, 32, layers[0], head_conv=3)

        self.fast_maxpool2 = nn.MaxPool3d(kernel_size=(2, 1, 1), stride=(2, 1, 1), padding=(0, 0, 0))

        self.fast_res3 = self._make_layer_fast(
            block, 64, layers[1], stride=2, head_conv=3)
        # self.fast_maxpool3 = nn.MaxPool3d(kernel_size=(1, 3, 3), stride=(1, 2, 2), padding=(0, 1, 1))
        self.fast_res4 = self._make_layer_fast(
            block, 128, layers[2], stride=2, head_conv=3)
        # self.fast_maxpool4 = nn.MaxPool3d(kernel_size=(1, 3, 3), stride=(1, 2, 2), padding=(0, 1, 1))
        # self.fast_res5 = self._make_layer_fast(
        #     block, 128, layers[3], stride=2, head_conv=3)
        # self.fast_maxpool5 = nn.MaxPool3d(kernel_size=(1, 3, 3), stride=(1, 2, 2), padding=(0, 1, 1))
        self.fast_avgpoool = nn.AvgPool3d(kernel_size=(1, 14, 14), stride=(1, 1, 1), padding=(0, 0, 0))
        # self.fast_dp = nn.Dropout(dropout)

        self.slow_inplanes = 64
        self.slow_conv1 = nn.Conv3d(3, 64, kernel_size=(1, 7, 7), stride=(1, 2, 2), padding=(0, 3, 3), bias=False)
        self.slow_bn1 = nn.BatchNorm3d(64)
        self.slow_relu = nn.ReLU(inplace=True)
        self.slow_maxpool = nn.MaxPool3d(kernel_size=(1, 3, 3), stride=(1, 2, 2), padding=(0, 1, 1))
        self.slow_res2 = self._make_layer_slow(block, 64, layers[0], head_conv=1)
        self.slow_res3 = self._make_layer_slow(
            block, 128, layers[1], stride=2, head_conv=1)
        self.slow_res4 = self._make_layer_slow(
            block, 256, layers[2], stride=2, head_conv=3)
        self.slow_res5 = self._make_layer_slow(
            block, 512, layers[3], stride=2, head_conv=3)
        self.slow_avepool = nn.AvgPool3d(kernel_size=(1, 7, 7), stride=(1, 1, 1), padding=(0, 0, 0))

        self.conv1x1_1 = nn.Conv3d(512, 1024, kernel_size=(1, 1, 1), stride=(1, 1, 1), padding=(0, 0, 0))
        self.bn_1 = nn.BatchNorm3d(1024)
        self.relu_1 = nn.ReLU(inplace=True)
        self.conv1x1_2 = nn.Conv3d(1024, 512, kernel_size=(1, 1, 1), stride=(1, 1, 1), padding=(0, 0, 0))
        self.bn_2 = nn.BatchNorm3d(512)
        self.relu_2 = nn.ReLU(inplace=True)
        self.vit = time_T(
            num_patches=16,
            num_classes=2,
            dim=512,
            depth=1,
            heads=8,
            mlp_dim=1024,
            pool='cls',
            dim_head=64,
            dropout=0.1,
            emb_dropout=0.1
        )
        # self.dp = nn.Dropout(dropout)
        # self.fc = nn.Linear(self.fast_inplanes + 2048, class_num, bias=False)

    def forward(self, input):
        spa = self.SpatPath(input[:, :, ::8, :, :])
        tem = self.TempPath(input)
        x = self.STfusion(spa, tem)
        x = self.conv1x1_1(x)
        x = self.bn_1(x)
        x = self.relu_1(x)
        x = self.conv1x1_2(x)
        x = self.bn_2(x)
        x = self.relu_2(x)
        # x = nn.AdaptiveAvgPool3d((16, 1, 1))(x)
        x = rearrange(x, 'b c t h w -> b (t h w) c')
        x = self.vit(x)
        return x

    def SpatPath(self,input):
        x = self.slow_conv1(input)
        x = self.slow_bn1(x)
        x = self.slow_relu(x)
        x = self.slow_maxpool(x)
        x = self.slow_res2(x)
        x = self.slow_res3(x)
        x = self.slow_res4(x)
        x = self.slow_res5(x)
        x = self.slow_avepool(x)
        return x

    def TempPath(self, input):
        x = self.fast_conv1(input)
        x = self.fast_bn1(x)
        x = self.fast_relu(x)
        pool1 = self.fast_maxpool(x)
        res2 = self.fast_res2(pool1)
        pool2 = self.fast_maxpool2(res2)
        res3 = self.fast_res3(pool2)
        # pool3 = self.fast_maxpool3(res3)
        res4 = self.fast_res4(res3)
        # pool4 = self.fast_maxpool4(res4)
        # res5 = self.fast_res5(res4)
        # pool5 = self.fast_maxpool5(res5)
        avepool = self.fast_avgpoool(res4)
        return avepool

    def STfusion(self, Spat, Temp):
        s = rearrange(Spat, 'b c t h w -> b (c t) h w')
        t = rearrange(Temp, 'b c t h w -> b (c t) h w')
        st = s + t
        st = torch.reshape(st, (-1, 512, 16, 1, 1))
        return st

    def _make_layer_fast(self, block, planes, blocks, stride=1, head_conv=1):
        downsample = None
        if stride != 1 or self.fast_inplanes != planes * block.expansion:
            downsample = nn.Sequential(
                nn.Conv3d(
                    self.fast_inplanes,
                    planes * block.expansion,
                    kernel_size=1,
                    stride=(1, stride, stride),
                    bias=False), nn.BatchNorm3d(planes * block.expansion))

        layers = []
        layers.append(block(self.fast_inplanes, planes, stride, downsample, head_conv=head_conv))
        self.fast_inplanes = planes * block.expansion
        for i in range(1, blocks):
            layers.append(block(self.fast_inplanes, planes, head_conv=head_conv))
        return nn.Sequential(*layers)

    def _make_layer_slow(self, block, planes, blocks, stride=1, head_conv=1):
        downsample = None
        if stride != 1 or self.slow_inplanes != planes * block.expansion:
            downsample = nn.Sequential(
                nn.Conv3d(
                    self.slow_inplanes,
                    planes * block.expansion,
                    kernel_size=1,
                    stride=(1, stride, stride),
                    bias=False), nn.BatchNorm3d(planes * block.expansion))

        layers = []
        layers.append(block(self.slow_inplanes, planes, stride, downsample, head_conv=head_conv))
        self.slow_inplanes = planes * block.expansion
        for i in range(1, blocks):
            layers.append(block(self.slow_inplanes, planes, head_conv=head_conv))

        self.slow_inplanes = planes * block.expansion
        return nn.Sequential(*layers)



if __name__ == "__main__":

    input_tensor = torch.autograd.Variable(torch.rand(1, 3, 32, 224, 224))
    model = STFusion(num_classes=2)
    output = model(input_tensor)
    print(output.size())