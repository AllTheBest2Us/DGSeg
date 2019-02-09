import torch
from torch import nn
from torchvision import models
import torch.nn.functional as F
from utils import get_upsampling_weight

#FIXME
# Downloaded from: https://download.pytorch.org/models/vgg16-397923af.pth
model_path = "./models/pretrained/vgg16-397923af.pth"

avg_size = (4, 4)

class FCN8s(nn.Module):
    def __init__(self, num_classes, pretrained=True, m_path=model_path):
        super(FCN8s, self).__init__()
        vgg = models.vgg16()

        if pretrained:
            vgg.load_state_dict(torch.load(m_path))
            print(m_path + " Loaded!!!")

        features, classifier = list(vgg.features.children()), list(vgg.classifier.children())

        features[0].padding = (100, 100)

        for f in features:
            if "MaxPool" in f.__class__.__name__:
                f.ceil_mode = True
            elif "ReLU" in f.__class__.__name__:
                f.inplace = True

        self.features1 = nn.Sequential(*features[:5])
        self.features3 = nn.Sequential(*features[5:17])
        self.features4 = nn.Sequential(*features[17:24])
        self.features5 = nn.Sequential(*features[24:])

        self.score_pool3 = nn.Conv2d(256, num_classes, kernel_size=1)
        self.score_pool4 = nn.Conv2d(512, num_classes, kernel_size=1)
        
        self.score_pool3.weight.data.zero_()
        self.score_pool3.bias.data.zero_()
        self.score_pool4.weight.data.zero_()
        self.score_pool4.bias.data.zero_()


        fc6 = nn.Conv2d(512, 4096, kernel_size=7)
        fc6.weight.data.copy_(classifier[0].weight.data.view(4096, 512, 7, 7))
        fc6.bias.data.copy_(classifier[0].bias.data)

        fc7 = nn.Conv2d(4096, 4096, kernel_size=1)
        fc7.weight.data.copy_(classifier[3].weight.data.view(4096, 4096, 1, 1))
        fc7.bias.data.copy_(classifier[3].bias.data)
        
        score_fr = nn.Conv2d(4096, num_classes, kernel_size=1)
        score_fr.weight.data.zero_()
        score_fr.bias.data.zero_()

        self.score_fr = nn.Sequential(
            fc6, nn.ReLU(inplace=True), nn.Dropout(), fc7, nn.ReLU(inplace=True), nn.Dropout(), score_fr
        )

        self.upscore2 = nn.ConvTranspose2d(num_classes, num_classes, kernel_size=4, stride=2, bias=False)
        self.upscore_pool4 = nn.ConvTranspose2d(num_classes, num_classes, kernel_size=4, stride=2, bias=False)
        self.upscore8 = nn.ConvTranspose2d(num_classes, num_classes, kernel_size=16, stride=8, bias=False)

        self.upscore2.weight.data.copy_(get_upsampling_weight(num_classes, num_classes, 4))
        self.upscore_pool4.weight.data.copy_(get_upsampling_weight(num_classes, num_classes, 4))
        self.upscore8.weight.data.copy_(get_upsampling_weight(num_classes, num_classes, 16))
        self.softmax = nn.LogSoftmax()
    
    def forward(self, x):
        x_size = x.size()
        result = []
        pool1 = self.features1(x)
        # print("pool1:", pool1.shape)
        pool3 = self.features3(pool1)
        # print("pool3:", pool3.shape)
        pool4 = self.features4(pool3)
        # print("pool4:", pool4.shape)
        pool5 = self.features5(pool4)
        # print("pool5:", pool5.shape)
        # result.append(F.adaptive_avg_pool2d(pool5, avg_size))

        score_fr = self.score_fr(pool5)
        # print("score_fr:", score_fr.shape)
        result.append(F.adaptive_avg_pool2d(score_fr, avg_size))

        upscore2 = self.upscore2(score_fr)
        # print("upscore2:", upscore2.shape)
        result.append(F.adaptive_avg_pool2d(upscore2, avg_size))

        score_pool4 = self.score_pool4(pool4)[:, :, 5:(5 + upscore2.size()[2]), 5:(5 + upscore2.size()[3])]
        # print("score_pool4:", score_pool4.shape)
        # result.append(F.adaptive_avg_pool2d(score_pool4, avg_size))

        upscore_pool4 = self.upscore_pool4(score_pool4
                                            + upscore2)
        # print("upscore_pool4:", upscore_pool4.shape)
        result.append(F.adaptive_avg_pool2d(upscore_pool4, avg_size))

        score_pool3 = self.score_pool3(pool3)[:, :, 9:(9+upscore_pool4.size()[2]), 9:(9+upscore_pool4.size()[3])]
        # print("score_pool3:", score_pool3.shape)
        result.append(F.adaptive_avg_pool2d(score_pool3, avg_size))

        upscore8 = self.upscore8(score_pool3
                                + upscore_pool4)
        # print("upscore8:", upscore8.shape)
        # result.append(F.adaptive_avg_pool2d(upscore8, avg_size))

        upscore8_cropped = upscore8[:, :, 31:(31 + x_size[2]), 31:(31+x_size[3])]
        result.append(F.adaptive_avg_pool2d(upscore8_cropped, avg_size))
        result.append(upscore8_cropped)

        return result, self.softmax(upscore8_cropped.contiguous())
