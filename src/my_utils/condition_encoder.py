import torch
import torch.nn as nn
import torch.nn.functional as F
import math

class SoftHistogram(nn.Module):

    def __init__(self, bins=64, min_val=0.0, max_val=1.0, sigma=0.01):
        super().__init__()
        self.bins = bins
        self.min_val = min_val
        self.max_val = max_val
        self.sigma = sigma

        self.register_buffer('bin_centers', torch.linspace(min_val, max_val, bins))

    def forward(self, x):

        b, c, h, w = x.shape
        x_flat = x.view(b, c, -1, 1)
        
        centers = self.bin_centers.view(1, 1, 1, self.bins)
        
        x_minus_mu = x_flat - centers
        weights = torch.exp(-torch.pow(x_minus_mu, 2) / (2 * self.sigma ** 2))
        

        hist = torch.sum(weights, dim=2)
        
        hist = hist / (hist.sum(dim=2, keepdim=True) + 1e-6)
        
        return hist

class ColorStatisticEncoder(nn.Module):

    def __init__(self, in_channels=3, hidden_dim=64):
        super().__init__()

        self.soft_hist = SoftHistogram(bins=32) # 每个通道32个bin

        self.color_mlp = nn.Sequential(
            nn.Conv2d(in_channels, hidden_dim, kernel_size=1),
            nn.GELU(),
            nn.Conv2d(hidden_dim, hidden_dim, kernel_size=1),
            nn.AdaptiveAvgPool2d(1) # 全局平均
        )
        
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
        
        hist_feat = self.soft_hist(x).view(b, -1)
        
        # [B, C, H, W] -> [B, C]
        feat_var, feat_mean = torch.var_mean(x, dim=(2, 3))
        feat_std = torch.sqrt(feat_var + 1e-6)
        stats_feat = torch.cat([feat_mean, feat_std], dim=1) # [B, C*2]
        
        color_feat = self.color_mlp(x).view(b, -1) # [B, hidden_dim]
        
        combined = torch.cat([hist_feat, stats_feat, color_feat], dim=1)
        
        return self.fusion(combined)

class HistogramLikeEncoder(nn.Module):
    def __init__(self, output_dim=1024, cond_len=1, dropout_rate=0.3):
        super().__init__()
        self.output_dim = output_dim
        self.cond_len = cond_len
        
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

        self.color_encoder = ColorStatisticEncoder(in_channels=3)

        self.fc_layers = nn.Sequential(

            nn.Linear(2048 + 256, 1024),
            nn.GELU(),
            nn.Dropout(dropout_rate),
            nn.Linear(1024, self.cond_len * self.output_dim),
            nn.LayerNorm(self.cond_len * self.output_dim)
        )
        
        self._initialize_weights()

    def _initialize_weights(self):

        for m in self.modules():
            if isinstance(m, nn.Conv2d):
                nn.init.kaiming_normal_(m.weight, mode='fan_out', nonlinearity='relu')
            elif isinstance(m, nn.Linear):
                nn.init.trunc_normal_(m.weight, std=0.02)

    def forward(self, x):
        # x: [B, 3, H, W] in [-1, 1]
        
        x_norm = (x + 1.0) / 2.0 
        
        spatial_feat = self.spatial_encoder(x) # [B, 128, 4, 4]
        spatial_feat = spatial_feat.view(spatial_feat.size(0), -1) # [B, 2048]
        
        color_feat = self.color_encoder(x_norm) # [B, 256]
        
        concat_feat = torch.cat([spatial_feat, color_feat], dim=1) # [B, 2304]
        
        out = self.fc_layers(concat_feat)
        condition = out.view(out.size(0), self.cond_len, self.output_dim)
        
        return condition
