import time
import torch
import torch.nn as nn
import torch.nn.functional as F
import torch
import torch.nn as nn
import torch.nn.functional as F
FUSION_MODEL=r"/opt/FSCE-main/resnet.pth"


class WeightedAttentionFusion(nn.Module):
    def __init__(self, input_dim):
        super(WeightedAttentionFusion, self).__init__()
        # 初始化权重参数为标量
        self.attn_weight_box = nn.Parameter(torch.tensor(0.5), requires_grad=True)
        self.attn_weight_fused = nn.Parameter(torch.tensor(0.5), requires_grad=True)
        
        # 全连接层，用于拼接后的特征融合
        self.fc = nn.Linear(input_dim * 2, input_dim)  # 拼接后输入维度增加一倍

    def forward(self, box_features, fused_features):
        # 将权重合并并进行 softmax 计算
        # total_weights = torch.cat([self.attn_weight_box.view(1), self.attn_weight_fused.view(1)])
        # attn_weights = F.softmax(total_weights, dim=0)
        
        # attn_box = attn_weights[0]
        # attn_fused = attn_weights[1]
        
        # 加权特征
        weighted_box = self.attn_weight_box * box_features
        weighted_fused = self.attn_weight_fused * fused_features
        
        # 拼接特征
        combined_features = torch.cat((weighted_box, weighted_fused), dim=1)
        
        # 使用全连接层融合
        batch_size, channels, height, width = combined_features.shape
        combined_features_flat = combined_features.view(batch_size, -1, height * width).permute(0, 2, 1)
        fused_output = self.fc(combined_features_flat).permute(0, 2, 1).view(batch_size, -1, height, width)
        
        # 残差连接
        output = box_features + fused_output
        
        return output


# class WeightedAttentionFusion(nn.Module):
#     def __init__(self, input_dim):
#         super(WeightedAttentionFusion, self).__init__()
#         # 初始化权重参数，形状为 1
#         # 预先定义数值
#         initial_weight_box = torch.full((1, input_dim, 1, 1), 0.6)  # 初始化为 0.8
#         initial_weight_fused = torch.full((1, input_dim, 1, 1), 0.4)  # 初始化为 0.2

#         # 将指定的初始数值赋值给参数
#         self.attn_weight_box = nn.Parameter(initial_weight_box,requires_grad=True)
#         self.attn_weight_fused = nn.Parameter(initial_weight_fused,requires_grad=True)

#         # self.attn_weight_box = nn.Parameter(torch.randn(1, input_dim, 1, 1))
#         # self.attn_weight_fused = nn.Parameter(torch.randn(1, input_dim, 1, 1))
        
#         # 全连接层，用于拼接后的特征融合
#         self.fc = nn.Linear(input_dim * 2, input_dim)  # 拼接后输入维度增加一倍
#         self.rate1 = nn.Parameter()

#     def forward(self, box_features, fused_features):
#         # 计算注意力权重，分别对 box_features 和 fused_features 加权
#         # attn_box = F.softmax(self.attn_weight_box, dim=1)
#         # attn_fused = F.softmax(self.attn_weight_fused, dim=1)
        
        
#         total_weights =torch.nn.Parameter(torch.cat([self.attn_weight_box, self.attn_weight_fused], dim=0),requires_grad=True)
#         attn_weights = F.softmax(total_weights, dim=0)
#         sattn_box = attn_weights[0]
#         attn_fused = attn_weights[1]
        
        

  
#         # 加权特征
#         weighted_box =attn_box * box_features
#         weighted_fused = attn_fused* fused_features
#         # print("attn_box:",torch.mean(weighted_box),"---"*3,"attn_fused:",torch.mean(weighted_fused))
#         # 拼接特征，沿着通道维度 (dim=1) 进行拼接
#         combined_features = torch.cat((weighted_box, weighted_fused), dim=1)  # 拼接后的特征维度变为原来的两倍
        
#         # 使用全连接层融合拼接后的特征，将形状从 [batch_size, channels*2, height, width] 转换为 [batch_size, channels, height, width]
#         # 先调整维度顺序
#         batch_size, channels, height, width = combined_features.shape
#         combined_features_flat = combined_features.view(batch_size, -1, height * width).permute(0, 2, 1)
#         fused_output = self.fc(combined_features_flat).permute(0, 2, 1).view(batch_size, -1, height, width)

#         # 将融合的特征与原始 box_features 相加（残差连接）
#         output = box_features + fused_output
        
#         return output


class TSFigureBackBone(nn.Module):
    def __init__(self, in_channels_list, out_channels=256):
        """
        初始化时空轨迹特征提取骨干网络。
        参数:
        - in_channels_list: 每个特征图的输入通道数列表，例如 [64, 256, 512, 1024]。
        - out_channels: 每个特征图经过卷积后的输出通道数。
        """
        super(TSFigureBackBone, self).__init__()
        
        # 定义卷积层来调整通道数
        self.convs = nn.ModuleList([
            nn.Conv2d(in_channels, out_channels, kernel_size=1) for in_channels in in_channels_list
        ])
        
        # 定义特征融合模块
        self.fusion = WeightedAttentionFusion(out_channels)

    def forward(self, features, src_feat=None):
        """
        前向传播函数，对多层次特征图进行通道调整和插值操作。
        
        参数：
        - features: 一个列表，包含多层次特征图。
        - src_feat: 额外的输入特征，可与多层次特征融合。

        返回：
        - 融合后的特征图，形状为 [batch_size, out_channels, ...]。
        """
        if src_feat is None:
            raise ValueError("src_feat None")
        
        out = {}
        
        # 遍历特征图及其对应的卷积层
        for i, (conv, feat, src) in enumerate(zip(self.convs, features, src_feat)):
            # 进行卷积和插值
            feat_out = conv(feat)
            feat_out = F.interpolate(feat_out, size=src_feat[src].shape[2:], mode='bilinear', align_corners=False)
            feat = self.fusion(fused_features = feat_out, box_features = src_feat[src])
            name = f"p{(i+2)}"
            out[name] = feat
            
            
            # if i == 0:  # 对第一个特征进行融合
            #     out.append(self.fusion(feat_out, src))
            # else:
            #     out.append(feat_out)

        return out
# 测试用例

class Fusion(nn.Module):
      def __init__(self,st_fig_channal_list=[64, 256, 512, 1024,2048], out_channels=256) -> None:
          super(Fusion,self).__init__()
          self.model = torch.load(FUSION_MODEL).eval()
          self.model = self.model.cuda()
          self.fusionNet = TSFigureBackBone(st_fig_channal_list, out_channels=out_channels).cuda()
      def forward(self,st_fig,src):#st图像
          features = self.model(st_fig)
          out = self.fusionNet(features,src)
          return out


import torch
from PIL import Image
from torchvision import transforms

def images_to_tensor(image_paths):
    # 定义转换：将图像转换为张量并归一化
    transform = transforms.Compose([
        transforms.Resize((256, 256)),  # 缩放图像大小到 256x256
        transforms.ToTensor()           # 转换为张量
    ])
    
    tensors = []
    for image_path in image_paths:
        # 读取并转换每个图像
        image = Image.open(image_path).convert("RGB")
        image_tensor = transform(image)
        tensors.append(image_tensor)
        
    stacked_tensors = torch.stack(tensors)
    return stacked_tensors

TEST_FUSION_TENSOR = torch.rand(2, 3, 256, 256).cuda()
'''


model = Fusion(src=features)
model(test)
'''