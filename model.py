import torch.nn as nn
import torch
import torch.nn.functional as F
from backbone import ResNet
from config import HyperParams
from thop import profile
from pytorch_model_summary import summary


class TopkPool(nn.Module):
    def __init__(self):
        super(TopkPool, self).__init__()

    def forward(self, x):
        b, c, _, _ = x.shape
        x = x.view(b, c, -1)
        topkv, _ = x.topk(5, dim=-1)        #按照指定的维度进行数值大小的排序，返回top-k个数值。
        return topkv.mean(dim=-1)

class BasicConv(nn.Module):
    def __init__(self, in_planes, out_planes, kernel_size, stride=1, padding=0, dilation=1, groups=1, relu=True,
                 bn=True, bias=False):
        super(BasicConv, self).__init__()
        self.out_channels = out_planes
        self.conv = nn.Conv2d(in_planes, out_planes, kernel_size=kernel_size,
                              stride=stride, padding=padding, dilation=dilation, groups=groups, bias=bias)
        self.bn = nn.BatchNorm2d(out_planes, eps=1e-5,
                                 momentum=0.01, affine=True) if bn else None
        self.relu = nn.ReLU() if relu else None

    def forward(self, x):
        x = self.conv(x)
        if self.bn is not None:
            x = self.bn(x)
        if self.relu is not None:
            x = self.relu(x)
        return x

class FDM(nn.Module):
    def __init__(self):
        super(FDM, self).__init__()
        self.factor = round(1.0/(28*28), 3)     #0.001

    def forward(self, fm1, fm2, fm3):
        b, c, w1, h1 = fm1.shape
        _, _, w2, h2 = fm2.shape
        _, _, w3, h3 = fm3.shape
        fm1_0 = fm1.view(b, c, -1) # B*C*S
        fm2 = fm2.view(b, c, -1) # B*C*M
        fm3 = fm3.view(b, c, -1)  # B*C*M
        fm1_t = fm1_0.permute(0, 2, 1) # B*S*C

        # may not need to normalize
        fm1_t_norm = F.normalize(fm1_t, dim=-1)
        fm2_norm = F.normalize(fm2, dim=1)
        fm3_norm = F.normalize(fm3, dim=1)
        M = -1 * torch.bmm(fm1_t_norm, fm2_norm) # B*S*M
        M2 = -1 * torch.bmm(fm1_t_norm, fm3_norm)
        #计算两个tensor的矩阵乘法，torch.bmm(a,b),tensor a 的size为(b,h,w),tensor b的size为(b,w,m) 也就是说两个tensor的第一维是相等的，然后第一个数组的第三维和第二个数组的第二维度要求一样，对于剩下的则不做要求，输出维度 （b,h,m）


        # M_1 = F.softmax(M, dim=1)
        M_2 = F.softmax(M.permute(0, 2, 1), dim=1)
        M_3 = F.softmax(M2.permute(0, 2, 1), dim=1)
        # new_fm2 = torch.bmm(fm1, M_1).view(b, c, w2, h2)        #相当于numpy中resize（）的功能，
        new_fm2 = torch.bmm(fm2, M_2).view(b, c, w1, h1)
        # print(new_fm2.shape)
        new_fm3 = torch.bmm(fm3, M_3).view(b, c, w1, h1)
        # print(new_fm3.shape)
        # print(fm1.shape)
        f1 = fm1 + self.factor*new_fm2 + self.factor*new_fm3
        # f2 = fm2 + self.factor*new_fm2
        # f1 = fm1 + new_fm2 + new_fm3

        return f1

class FBSD(nn.Module):
    def __init__(self, class_num, arch='resnet50'):
        super(FBSD, self).__init__()
        feature_size = 512
        if arch == 'resnet50':
            self.features = ResNet(arch='resnet50')
            chans = [512, 1024, 2048]
        elif arch == 'resnet101':
            self.features = ResNet(arch='resnet101')
            chans = [512, 1024, 2048]


        self.pool = TopkPool()

        part_feature = 1024

        self.classifier_concat = nn.Sequential(
            nn.BatchNorm1d(part_feature* 6),
            nn.Linear(part_feature* 6, feature_size),
            nn.BatchNorm1d(feature_size),
            nn.ELU(inplace=True),
            nn.Linear(feature_size, class_num)
        )
        self.conv_block1 = nn.Sequential(
            BasicConv(chans[0], feature_size, kernel_size=1, stride=1, padding=0, relu=True),
            BasicConv(feature_size, part_feature, kernel_size=3, stride=1, padding=1, relu=True),
        )
        self.classifier1 = nn.Sequential(

            nn.BatchNorm1d(part_feature),
            nn.Linear(part_feature, feature_size),
            nn.BatchNorm1d(feature_size),
            nn.ELU(inplace=True),
            nn.Linear(feature_size, class_num)
        )
        self.conv_block2 = nn.Sequential(
            BasicConv(chans[1], feature_size, kernel_size=1, stride=1, padding=0, relu=True),
            BasicConv(feature_size, part_feature, kernel_size=3, stride=1, padding=1, relu=True),
        )
        self.classifier2 = nn.Sequential(
            nn.BatchNorm1d(part_feature),
            nn.Linear(part_feature, feature_size),
            nn.BatchNorm1d(feature_size),
            nn.ELU(inplace=True),
            nn.Linear(feature_size, class_num)
        )

        self.conv_block3 = nn.Sequential(
            BasicConv(chans[2], feature_size, kernel_size=1, stride=1, padding=0, relu=True),
            BasicConv(feature_size, part_feature, kernel_size=3, stride=1, padding=1, relu=True),
        )
        self.classifier3 = nn.Sequential(
            nn.BatchNorm1d(part_feature),
            nn.Linear(part_feature, feature_size),
            nn.BatchNorm1d(feature_size),
            nn.ELU(inplace=True),
            nn.Linear(feature_size, class_num)
        )

        self.conv_block4 = nn.Sequential(
            BasicConv(chans[0], feature_size, kernel_size=1, stride=1, padding=0, relu=True),
            BasicConv(feature_size, part_feature, kernel_size=3, stride=1, padding=1, relu=True),
        )
        self.classifier4 = nn.Sequential(
            nn.BatchNorm1d(part_feature),
            nn.Linear(part_feature, feature_size),
            nn.BatchNorm1d(feature_size),
            nn.ELU(inplace=True),
            nn.Linear(feature_size, class_num)
        )
        self.conv_block5 = nn.Sequential(
            BasicConv(chans[1], feature_size, kernel_size=1, stride=1, padding=0, relu=True),
            BasicConv(feature_size, part_feature, kernel_size=3, stride=1, padding=1, relu=True),
        )
        self.classifier5 = nn.Sequential(
            nn.BatchNorm1d(part_feature),
            nn.Linear(part_feature, feature_size),
            nn.BatchNorm1d(feature_size),
            nn.ELU(inplace=True),
            nn.Linear(feature_size, class_num)
        )

        self.conv_block6 = nn.Sequential(
            BasicConv(chans[2], feature_size, kernel_size=1, stride=1, padding=0, relu=True),
            BasicConv(feature_size, part_feature, kernel_size=3, stride=1, padding=1, relu=True),
        )
        self.classifier6 = nn.Sequential(
            nn.BatchNorm1d(part_feature),
            nn.Linear(part_feature, feature_size),
            nn.BatchNorm1d(feature_size),
            nn.ELU(inplace=True),
            nn.Linear(feature_size, class_num)
        )

        self.inter1 = FDM()
        self.inter2 = FDM()
        self.inter3 = FDM()
        self.inter4 = FDM()
        self.inter5 = FDM()
        self.inter6 = FDM()

    def forward(self, x):
        fm1, fm2, fm3, drop1, drop2, drop3 = self.features(x)
        # fm1就表示FBSM输出的显著性特征图，即Xb1

        #########################################
        ##### cross-level attention #############
        #########################################
        #通道注意力图att1, att2, att3
        att1 = self.conv_block1(fm1)
        att2 = self.conv_block2(fm2)
        att3 = self.conv_block3(fm3)
        #
        att4 = self.conv_block4(drop1)
        att5 = self.conv_block5(drop2)
        att6 = self.conv_block6(drop3)
        #将得到的通道注意力送入FDM模块（进行特征的融合，即多样化处理），注意注意力向量两两结合的方式，1和2、1和3、2和3。最后可以得到两张注意图相互之间互补的信息.
        # new_d1_from2, new_d2_from1 = self.inter(att1, att2)  # 1 2
        # new_d1_from3, new_d3_from1 = self.inter(att1, att3)  # 1 3
        # new_d2_from3, new_d3_from2 = self.inter(att2, att3)  # 2 3
        # new_d4_from5, new_d5_from4 = self.inter(att4, att5)  # 1 2
        # new_d4_from6, new_d6_from4 = self.inter(att4, att6)  # 1 3
        # new_d5_from6, new_d6_from5 = self.inter(att5, att6)  # 2 3
        att1 = self.inter1(att1, att5, att6)
        att2 = self.inter2(att2, att4, att6)  # 1 6
        att3 = self.inter3(att3, att4, att5)  # 2 4
        att4 = self.inter4(att4, att2, att3)  # 1 2
        att5 = self.inter5(att5, att1, att6)  # 3 4
        att6 = self.inter6(att6, att1, att2)  # 3 5
        #把刚才得到的互补的信息进行融合得到增强后的特征
        # gamma = HyperParams['gamma']
        # att1 = att1 + gamma*(new_d1_from5 + new_d1_from6)
        # att2 = att2 + gamma*(new_d2_from4 + new_d2_from6)
        # att3 = att3 + gamma*(new_d3_from4 + new_d3_from5)
        # att4 = att4 + gamma * (new_d4_from2 + new_d4_from3)
        # att5 = att5 + gamma * (new_d5_from1 + new_d5_from3)
        # att6 = att6 + gamma * (new_d6_from1 + new_d6_from2)
        #对上一步得到的增强后的特征图进行分层池化(TopK)
        xl1 = self.pool(att1)
        xc1 = self.classifier1(xl1)     #最后进行线性分类

        xl2 = self.pool(att2)
        xc2 = self.classifier2(xl2)

        xl3 = self.pool(att3)
        xc3 = self.classifier3(xl3)

        xl4 = self.pool(att4)
        xc4 = self.classifier4(xl4)  # 最后进行线性分类

        xl5 = self.pool(att5)
        xc5 = self.classifier5(xl5)
        #
        xl6 = self.pool(att6)
        xc6 = self.classifier6(xl6)

        x_concat = torch.cat((xl1, xl2, xl3, xl4, xl5, xl6), -1)
        x_concat = self.classifier_concat(x_concat)

        return xc1, xc2, xc3, xc4, xc5, xc6, x_concat

    def get_params(self):
        new_layers, old_layers = self.features.get_params()
        new_layers += list(self.conv_block1.parameters()) + \
                      list(self.conv_block2.parameters()) + \
                      list(self.conv_block3.parameters()) + \
                      list(self.classifier1.parameters()) + \
                      list(self.classifier2.parameters()) + \
                      list(self.classifier3.parameters()) + \
                      list(self.conv_block4.parameters()) + \
                      list(self.conv_block5.parameters()) + \
                      list(self.conv_block6.parameters()) + \
                      list(self.classifier4.parameters()) + \
                      list(self.classifier5.parameters()) + \
                      list(self.classifier6.parameters()) + \
                      list(self.inter1.parameters()) + \
                      list(self.inter2.parameters()) + \
                      list(self.inter3.parameters()) + \
                      list(self.inter4.parameters()) + \
                      list(self.inter5.parameters()) + \
                      list(self.inter6.parameters()) + \
                      list(self.classifier_concat.parameters())
        return new_layers, old_layers

if __name__ == '__main__':
    x = torch.randn((2,3, 448, 448))
    model = FBSD(class_num=200, arch='resnet50')
    xc1, xc2, xc3, xc4, xc5, xc6, x_concat = model(x)
    # print(fm2.shape, fm3.shape, fm4.shape)
    # print(model)
    # 计算模型参数量
    flops, params = profile(model, (x,))
    print('flops: ', flops, 'params: ', params)
    print('flops: %.2f M, params: %.2f M' % (flops / 1000000.0, params / 1000000.0))
    print(summary(model, x, show_input=False, show_hierarchical=False))
