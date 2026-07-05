import torch
import torch.nn as nn
import torch.nn.functional as F
import math

class SoftHistogram(nn.Module):
    """
    可微分直方图层
    通过径向基函数(RBF)模拟直方图的"分箱"操作，使其可反向传播。
    """
    def __init__(self, bins=64, min_val=0.0, max_val=1.0, sigma=0.01):
        super().__init__()
        self.bins = bins
        self.min_val = min_val
        self.max_val = max_val
        self.sigma = sigma
        # 固定这种bin的中心位置，不需要训练
        self.register_buffer('bin_centers', torch.linspace(min_val, max_val, bins))

    def forward(self, x):
        """
        Args:
            x: [B, C, H, W]
        Returns:
            hist: [B, C, bins]
        """
        # x: [B, C, H, W] -> [B, C, H*W, 1]
        b, c, h, w = x.shape
        x_flat = x.view(b, c, -1, 1)
        
        # centers: [bins] -> [1, 1, 1, bins]
        centers = self.bin_centers.view(1, 1, 1, self.bins)
        
        # 计算每个像素值与bin中心的距离 (高斯核)
        # result: [B, C, H*W, bins]
        x_minus_mu = x_flat - centers
        weights = torch.exp(-torch.pow(x_minus_mu, 2) / (2 * self.sigma ** 2))
        
        # 在空间维度(H*W)上求和，模拟直方图的"计数"
        # hist: [B, C, bins]
        hist = torch.sum(weights, dim=2)
        
        # 归一化 (可选，使总和为1)
        hist = hist / (hist.sum(dim=2, keepdim=True) + 1e-6)
        
        return hist

class ColorStatisticEncoder(nn.Module):
    """
    色彩统计专用编码器
    包含：软直方图 + 统计矩(均值/方差) + 1x1卷积色彩映射
    """
    def __init__(self, in_channels=3, hidden_dim=64):
        super().__init__()
        # 1. 软直方图特征 (显式分布)
        self.soft_hist = SoftHistogram(bins=32) # 每个通道32个bin
        
        # 2. 统计矩特征 (均值和标准差)
        # 不需要层结构，在forward中计算
        
        # 3. 逐像素色彩映射 (1x1 Conv) - 忽略空间结构，只看颜色映射
        self.color_mlp = nn.Sequential(
            nn.Conv2d(in_channels, hidden_dim, kernel_size=1),
            nn.GELU(),
            nn.Conv2d(hidden_dim, hidden_dim, kernel_size=1),
            nn.AdaptiveAvgPool2d(1) # 全局平均
        )
        
        # 融合层: 输入维度 = (通道数*bins) + (通道数*2 均值方差) + hidden_dim
        hist_dim = in_channels * 32
        stats_dim = in_channels * 2 
        total_in_dim = hist_dim + stats_dim + hidden_dim
        
        self.fusion = nn.Sequential(
            nn.Linear(total_in_dim, 256),
            nn.LayerNorm(256),
            nn.GELU()
        )

    def forward(self, x):
        b, c, h, w = x.shape
        
        # A. 提取直方图特征 [B, C, bins] -> Flatten -> [B, C*bins]
        hist_feat = self.soft_hist(x).view(b, -1)
        
        # B. 提取统计矩特征 (Mean & Std)
        # [B, C, H, W] -> [B, C]
        feat_var, feat_mean = torch.var_mean(x, dim=(2, 3))
        feat_std = torch.sqrt(feat_var + 1e-6)
        stats_feat = torch.cat([feat_mean, feat_std], dim=1) # [B, C*2]
        
        # C. 提取纯色彩语义
        color_feat = self.color_mlp(x).view(b, -1) # [B, hidden_dim]
        
        # D. 拼接
        combined = torch.cat([hist_feat, stats_feat, color_feat], dim=1)
        
        return self.fusion(combined)

class HistogramLikeEncoder(nn.Module):
    def __init__(self, output_dim=1024, cond_len=1, dropout_rate=0.3):
        super().__init__()
        self.output_dim = output_dim
        self.cond_len = cond_len
        
        # --- 分支1: 你的原始空间特征提取 (略微精简以平衡参数) ---
        self.spatial_encoder = nn.Sequential(
            nn.Conv2d(3, 32, 3, padding=1),
            nn.BatchNorm2d(32),
            nn.ReLU(inplace=True),
            nn.MaxPool2d(2),
            
            nn.Conv2d(32, 64, 3, padding=1),
            nn.BatchNorm2d(64),
            nn.ReLU(inplace=True),
            nn.MaxPool2d(2),
            
            nn.Conv2d(64, 128, 3, padding=1),
            nn.BatchNorm2d(128),
            nn.ReLU(inplace=True),
            nn.AdaptiveAvgPool2d((4, 4)) # [128, 4, 4]
        )
        # 空间特征展平维度: 128*4*4 = 2048
        
        # --- 分支2: 新增的色彩统计编码器 ---
        self.color_encoder = ColorStatisticEncoder(in_channels=3)
        # 输出维度: 256
        
        # --- 最终融合与映射 ---
        self.fc_layers = nn.Sequential(
            # 输入维度 = 空间特征(2048) + 色彩特征(256)
            nn.Linear(2048 + 256, 1024),
            nn.GELU(),
            nn.Dropout(dropout_rate),
            nn.Linear(1024, self.cond_len * self.output_dim),
            nn.LayerNorm(self.cond_len * self.output_dim)
        )
        
        self._initialize_weights()

    def _initialize_weights(self):
        # 保持你的初始化逻辑
        for m in self.modules():
            if isinstance(m, nn.Conv2d):
                nn.init.kaiming_normal_(m.weight, mode='fan_out', nonlinearity='relu')
            elif isinstance(m, nn.Linear):
                nn.init.trunc_normal_(m.weight, std=0.02)

    def forward(self, x):
        # x: [B, 3, H, W] in [-1, 1]
        
        # 1. 预处理
        # 空间特征提取用 [-1, 1] 没问题
        # 直方图统计最好用 [0, 1]，方便固定bin范围
        x_norm = (x + 1.0) / 2.0 
        
        # 2. 分支处理
        # 分支A: 空间结构特征
        spatial_feat = self.spatial_encoder(x) # [B, 128, 4, 4]
        spatial_feat = spatial_feat.view(spatial_feat.size(0), -1) # [B, 2048]
        
        # 分支B: 色彩统计特征 (使用归一化后的图像)
        color_feat = self.color_encoder(x_norm) # [B, 256]
        
        # 3. 特征拼接
        concat_feat = torch.cat([spatial_feat, color_feat], dim=1) # [B, 2304]
        
        # 4. 映射输出
        out = self.fc_layers(concat_feat)
        condition = out.view(out.size(0), self.cond_len, self.output_dim)
        
        return condition

# # 验证代码
# if __name__ == "__main__":
#     device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
#     input_image = torch.randn(2, 3, 256, 256).to(device)
#     encoder = EnhancedHistogramLikeEncoder().to(device)
#     output = encoder(input_image)
#     print(f"输入: {input_image.shape}")
#     print(f"输出: {output.shape}") 
#     # 验证梯度传播
#     loss = output.sum()
#     loss.backward()
#     print("反向传播成功，SoftHistogram 可训练。")