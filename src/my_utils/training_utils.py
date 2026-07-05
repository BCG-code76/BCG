import os
import random
import argparse
import json
import torch
from PIL import Image
from torchvision import transforms
import torchvision.transforms.functional as F
from glob import glob
import numpy as np

import os


class HistogramLayers(object):
    """
    NumPy 实现的直方图层基类：
    - 构建 1D 激活图（近似直方图）
    - 计算地球移动距离（EMD）损失、条件熵损失
    注：输入图像需为 NumPy 数组，形状遵循 [batch_size, H, W]（单通道）或 [batch_size, H, W, C]（多通道，需单通道输入）
    """

    def __init__(self, src, out, tgt, args):
        """
        初始化：计算输入数据的激活图，并保存配置参数
        :param out: 模型输出图像，形状 [batch_size, H, W]（单通道）
        :param tgt: 目标图像，形状 [batch_size, H, W]（单通道）
        :param args: 配置参数对象，需包含 bin_num、min_val、max_val、kernel_width_ratio
        """
        # 1. 保存配置参数
        self.bin_num = args.bin_num          # 直方图分箱数
        self.min_val = torch.tensor(args.min_val, device=out.device, dtype=out.dtype)
        self.max_val = torch.tensor(args.max_val, device=out.device, dtype=out.dtype)
        self.interval_length = (self.max_val - self.min_val) / self.bin_num  # 每个分箱的区间长度
        self.kernel_width = self.interval_length / args.kernel_width_ratio  # 激活函数核宽度

        # 2. 计算输入数据的激活图（核心步骤，替代原 TF 的 calc_activation_maps）
        self.maps_src = self.calc_activation_maps(src)  # 源图像的激活图
        self.maps_out = self.calc_activation_maps(out)  # 输出图像的激活图
        self.maps_tgt = self.calc_activation_maps(tgt)  # 目标图像的激活图

        # 3. 推导基础维度（batch size、像素总数）
        self.bs = self.maps_out.shape[0]                # batch size（批次大小）
        self.n_pixels = self.maps_out.shape[1]          # 单张图像总像素数（H*W）

    def calc_activation_maps(self, img):
        """
        计算图像的激活图（近似直方图的连续表示）
        :param img: 输入图像，形状 [batch_size, H, W]（单通道）
        :return maps: 激活图，形状 [batch_size, H*W, bin_num]
        """
        # 步骤1：生成分箱的中心点（如 bin_num=100，则生成 100 个分箱中心）
        bins_min_max = torch.linspace(
            self.min_val.item(), self.max_val.item(), self.bin_num + 1,
            device=img.device, dtype=img.dtype
        )
        bins_av = (bins_min_max[:-1] + bins_min_max[1:]) / 2  # [bin_num]
        bins_av = bins_av.unsqueeze(0).unsqueeze(0)  # [1, 1, bin_num]

        # 2. 图像展平 [batch_size, H*W, 1]
        img_flat = img.reshape(img.shape[0], -1)  # [batch_size, H*W]
        img_flat = img_flat.unsqueeze(-1)  # 扩展为 [batch_size, H*W, 1]

        # 步骤3：计算激活图（Sigmoid 组合函数，与原 TF 逻辑完全一致）
        maps = self.activation_func(img_flat, bins_av)
        return maps

    def activation_func(self, img_flat, bins_av):
        """
        激活函数：用 Sigmoid 组合近似“移位矩形函数”，生成每个像素在各分箱的响应
        :param img_flat: 展平后的图像，形状 [batch_size, H*W, 1]
        :param bins_av: 分箱中心点，形状 [1, 1, bin_num]
        :return: 激活响应，形状 [batch_size, H*W, bin_num]
        """
        # 基础计算（利用 NumPy 广播特性，自动匹配 batch 和像素维度）
        img_minus_bins_av = img_flat - bins_av  # [batch_size, H*W, bin_num]
        img_plus_bins_av = img_flat + bins_av   # [batch_size, H*W, bin_num]

        # Sigmoid 组合（与原 TF 公式完全对齐）
        term1 = torch.sigmoid((img_minus_bins_av + self.interval_length / 2) / self.kernel_width)
        term2 = torch.sigmoid((img_minus_bins_av - self.interval_length / 2) / self.kernel_width)
        term3 = torch.sigmoid((img_plus_bins_av - 2 * self.min_val + self.interval_length / 2) / self.kernel_width)
        term4 = torch.sigmoid((img_plus_bins_av - 2 * self.min_val - self.interval_length / 2) / self.kernel_width)
        term5 = torch.sigmoid((img_plus_bins_av - 2 * self.max_val + self.interval_length / 2) / self.kernel_width)
        term6 = torch.sigmoid((img_plus_bins_av - 2 * self.max_val - self.interval_length / 2) / self.kernel_width)

        return term1 - term2 + term3 - term4 + term5 - term6

    def calc_cond_entropy_loss(self, maps_x, maps_y):
        """
        计算条件熵损失 H(X|Y) = H(Y) - H(X,Y)
        :param maps_x: X 的激活图，形状 [batch_size, H*W, bin_num]
        :param maps_y: Y 的激活图，形状 [batch_size, H*W, bin_num]
        :return mean_cond_entropy: 批次平均条件熵，标量
        """
        # 1. 计算联合概率 p(x,y) = (X^T · Y) / 总像素数（X^T 表示 maps_x 转置，维度 [batch_size, bin_num, H*W]）
        pxy = torch.matmul(maps_x.transpose(1, 2), maps_y) / self.n_pixels   # [batch_size, bin_num, bin_num]

        # 2. 计算边缘概率 p(y) = 对 p(x,y) 沿 x 维度求和
        py = torch.sum(pxy, dim=1)  # [batch_size, bin_num]

        # 3. 计算熵 H(Y) 和联合熵 H(X,Y)（处理 0*log(0) 为 0 的情况）
        def xlogy(x):
            """x * log(x)，x=0时返回0"""
            mask = x > 0
            res = torch.zeros_like(x)
            res[mask] = x[mask] * torch.log(x[mask])
            return res

        hy = torch.sum(xlogy(py), dim=1)          # H(Y) [batch_size]
        hxy = torch.sum(xlogy(pxy), dim=(1, 2))   # H(X,Y) [batch_size]

        # 4. 计算条件熵并求批次平均
        cond_entropy = hy - hxy  # [batch_size]
        mean_cond_entropy = torch.mean(cond_entropy)  # 标量
        return mean_cond_entropy

    def ecdf(self, maps):
        """
        计算累积分布函数（CDF）：对激活图的概率分布求和
        :param maps: 激活图，形状 [batch_size, H*W, bin_num]
        :return cdf: 累积分布函数，形状 [batch_size, bin_num]
        """
        # 1. 计算概率分布 p = 激活图沿像素维度求和 / 总像素数
        p = torch.sum(maps, axis=1) / self.n_pixels  # [batch_size, bin_num]
        # 2. 计算 CDF（沿分箱维度累加）
        cdf = torch.cumsum(p, axis=1)  # [batch_size, bin_num]
        return cdf

    def emd_loss(self, maps, maps_hat):
        """
        计算地球移动距离（EMD）损失：用 CDF 的 L2 距离近似
        :param maps: 目标激活图（如真实图像），形状 [batch_size, H*W, bin_num]
        :param maps_hat: 预测激活图（如模型输出），形状 [batch_size, H*W, bin_num]
        :return mean_emd: 批次平均 EMD 损失，标量
        """
        # 1. 计算两者的 CDF
        ecdf_p = self.ecdf(maps)      # 目标 CDF，[batch_size, bin_num]
        ecdf_p_hat = self.ecdf(maps_hat)  # 预测 CDF，[batch_size, bin_num]

        # 2. 计算 EMD（CDF 差的绝对值平方的均值开根号）
        abs_diff = torch.abs(ecdf_p - ecdf_p_hat)  # [batch_size, bin_num]
        squared_diff = torch.square(abs_diff)      # [batch_size, bin_num]
        mean_squared = torch.mean(squared_diff, axis=-1)  # 沿分箱维度求均值，[batch_size]
        batch_emd = torch.sqrt(mean_squared)       # 单样本 EMD，[batch_size]

        # 3. 批次平均 EMD
        mean_emd = torch.mean(batch_emd)  # 标量
        return mean_emd

    def calc_hist_loss_tar_out(self):
        """计算目标图像与输出图像的 EMD 损失（封装便捷方法）"""
        return self.emd_loss(self.maps_tgt, self.maps_out)

    def calc_cond_entropy_loss_tar_out(self):
        """计算目标图像与输出图像的条件熵损失（封装便捷方法）"""
        return self.calc_cond_entropy_loss(self.maps_tgt, self.maps_out)



def histogram_loss(src, out, tgt, args):


    src = rgb2hsi(src)
    out = rgb2hsi(out)
    tgt = rgb2hsi(tgt)

    # 校验：HSI转换后无异常（首次运行可保留）
    assert not torch.any(torch.isnan(torch.stack([src, out, tgt]))).item(), "HSI转换后出现NaN"

    # 2. 初始化直方图层（每个通道单独计算，增加激活图校验）
    def create_hist_layer(channel_src, channel_out, channel_tgt):
        hist = HistogramLayers(src=channel_src, out=channel_out, tgt=channel_tgt, args=args)
        # 校验：激活图不为全0（否则熵损失会为0）
        assert torch.all(hist.maps_out > 0).item(), "输出激活图存在0值，导致hist_loss=0"
        assert torch.all(hist.maps_tgt > 0).item(), "目标激活图存在0值，导致hist_loss=0"
        return hist

    hist_1 = HistogramLayers(src=src[..., 0], out=out[..., 0], tgt=tgt[..., 0], args=args)
    hist_2 = HistogramLayers(src=src[..., 1], out=out[..., 1], tgt=tgt[..., 1], args=args)
    hist_3 = HistogramLayers(src=src[..., 2], out=out[..., 2], tgt=tgt[..., 2], args=args)

    # 3. 计算条件熵损失（修复0*log(0)处理）
    def safe_cond_entropy(hist_layer):
        # 调用HistogramLayers的calc_cond_entropy_loss，但增加后处理
        entropy = hist_layer.calc_cond_entropy_loss_tar_out()
        # 处理可能的NaN（如熵计算结果为负或NaN）
        entropy = torch.clamp(entropy, min=0.0)  # 熵不可能为负
        assert not torch.isnan(entropy).item(), "条件熵计算出现NaN"
        return entropy

    #mi_loss
    mi_loss_1 = hist_1.calc_cond_entropy_loss_tar_out()
    mi_loss_2 = hist_2.calc_cond_entropy_loss_tar_out()
    mi_loss_3 = hist_3.calc_cond_entropy_loss_tar_out()
    entropy_loss = (mi_loss_1 + mi_loss_2 + mi_loss_3)/3


    # 4. 最终校验：损失值无异常
    assert not torch.isnan(entropy_loss).item(), "最终熵损失出现NaN"
    assert entropy_loss >= 0.0, f"熵损失为负：{entropy_loss.item()}"

    # return entropy_loss*args.lambda_entropy + emd_loss*args.lambda_emd
    return entropy_loss*args.lambda_entropy


def rgb2hsi(img):
    """
    重构版RGB转HSI：强化数值稳定性+中间值校验，输入输出范围[-1,1]
    """
    # -------------------------- 1. 维度处理（保持兼容，增加设备一致性） --------------------------
    if isinstance(img, Image.Image):
        img = np.array(img).astype(np.float32)
    # 统一转为PyTorch张量（若输入是numpy），确保计算设备一致
    is_numpy = isinstance(img, np.ndarray)
    if is_numpy:
        img = torch.from_numpy(img).float().to("cuda" if torch.cuda.is_available() else "cpu")
    # 维度转换：[B,C,H,W] / [C,H,W] → [B,H,W,3] / [H,W,3]
    original_shape = img.shape  # 保存原始形状，用于最终还原
    if img.ndim == 4 and img.shape[1] == 3:
        img_rgb = img.permute(0, 2, 3, 1).contiguous()  # B,C,H,W → B,H,W,3
    elif img.ndim == 3 and img.shape[0] == 3:
        img_rgb = img.permute(1, 2, 0).contiguous()  # C,H,W → H,W,3
    elif (img.ndim == 3 and img.shape[-1] == 3) or (img.ndim == 4 and img.shape[-1] == 3):
        img_rgb = img
    else:
        raise ValueError(f"不支持的输入维度：{original_shape}，仅支持 [B,3,H,W]、[3,H,W]、[H,W,3]")

    # -------------------------- 2. 数值范围稳定化（关键：避免极端值） --------------------------
    # 1. 若输入为[-1,1]，先转换为[0,1]；若输入已为[0,1]则直接使用
    # 判断输入范围（通过最大值粗略判断）
    if torch.max(img_rgb) <= 1.0 + 1e-6 and torch.min(img_rgb) >= -1.0 - 1e-6:
        img_rgb = (img_rgb + 1.0) / 2.0  # [-1,1] → [0,1]
    eps = torch.tensor(1e-6, device=img_rgb.device, dtype=img_rgb.dtype)
    # 2. 强制裁剪到 [eps, 1-eps]（彻底杜绝0和1，避免后续计算异常）
    img_rgb = torch.clamp(img_rgb, min=eps, max=1.0 - eps)
    # 校验：确保裁剪后无极端值（首次运行可保留，稳定后可注释）
    assert torch.all(img_rgb >= eps).item(), f"RGB最小值异常：{img_rgb.min().item()}"
    assert torch.all(img_rgb <= 1.0 - eps).item(), f"RGB最大值异常：{img_rgb.max().item()}"

    # -------------------------- 3. HSI核心计算（每步都加稳定性处理） --------------------------
    r, g, b = img_rgb[..., 0], img_rgb[..., 1], img_rgb[..., 2]

    # 3.1 强度I（简单平均，无风险）
    i = (r + g + b) / 3.0
    i = torch.clamp(i, 0.0, 1.0)  # 额外裁剪，确保I在[0,1]

    # 3.2 饱和度S（优化除法：用max替代softplus，避免梯度突变）
    sum_rgb = r + g + b  # 理论范围 [3*eps, 3*(1-eps)]
    min_rgb = torch.min(torch.stack([r, g, b], dim=-1), dim=-1)[0]
    # 分母用max(3*eps, sum_rgb)：确保分母不小于3*eps，比softplus更稳定
    sum_rgb_safe = torch.max(sum_rgb, torch.full_like(sum_rgb, 3 * eps))
    # 分子加eps：避免分子为0导致S=1（反向传播时梯度突变）
    s = 1.0 - (3.0 * (min_rgb + eps)) / sum_rgb_safe
    # 强制S在[0,1]（物理意义上饱和度不可能为负或大于1）
    s = torch.clamp(s, 0.0, 1.0)
    # 校验：S无异常（首次运行可保留）
    assert not torch.any(torch.isnan(s)).item(), "饱和度S出现NaN"

    # 3.3 色调H（重点：解决反余弦梯度爆炸）
    # 分子：0.5*((r-g)+(r-b)) → 等价于 r - 0.5*(g+b)，避免重复计算
    numerator = r - 0.5 * (g + b)
    # 分母：sqrt((r-g)² + (r-b)(g-b)) + eps² → 避免开方后为0
    denominator = torch.sqrt((r - g) ** 2 + (r - b) * (g - b) + eps ** 2)
    # 余弦值裁剪到 [-0.999, 0.999]：acos(x)在x→±1时梯度→±∞，远离边界！
    cos_theta = numerator / denominator
    cos_theta = torch.clamp(cos_theta, min=-0.999, max=0.999)
    # 计算theta：用torch.acos，配合裁剪后无梯度爆炸
    theta = torch.acos(cos_theta)
    # 处理B>G的情况：用where保持计算图连续
    h = torch.where(b > g, 2 * torch.pi - theta, theta)
    # 归一化到[0,1]：除以2π
    h = h / (2 * torch.pi)
    # 校验：H无异常（首次运行可保留）
    assert not torch.any(torch.isnan(h)).item(), "色调H出现NaN"

    # -------------------------- 4. 输出范围映射+强约束 --------------------------
    h = torch.clamp(h, 0.0, 1.0 - eps)  # H: [0,1)
    s = torch.clamp(s, 0.0, 1.0)        # S: [0,1]
    i = torch.clamp(i, 0.0, 1.0)        # I: [0,1]
    # 最终校验
    assert not torch.any(torch.isnan(torch.stack([h, s, i]))).item(), "HSI出现NaN"
    assert torch.all(h >= 0.0).item() and torch.all(h < 1.0).item(), f"H范围异常：{h.min().item()}~{h.max().item()}"
    assert torch.all(s >= 0.0).item() and torch.all(s <= 1.0).item(), f"S范围异常：{s.min().item()}~{s.max().item()}"
    assert torch.all(i >= 0.0).item() and torch.all(i <= 1.0).item(), f"I范围异常：{i.min().item()}~{i.max().item()}"


    # -------------------------- 5. 还原原始维度和类型 --------------------------
    # 堆叠HSI通道：[B,H,W,3] / [H,W,3]
    img_hsi = torch.stack([h, s, i], dim=-1).contiguous()
    # 若输入是numpy，转回numpy（保持兼容）
    if is_numpy:
        img_hsi = img_hsi.cpu().numpy()

    return img_hsi


def cmmd(x, y, sigma=10.0, scale=1000):
    """
    计算两个特征分布x和y的CMMD距离（基于内存高效的MMD实现修改）
    x: 形状为[batch_size, feature_dim]的张量
    y: 形状为[batch_size, feature_dim]的张量
    sigma: RBF核的带宽参数
    scale: 结果缩放因子，使 metric 更易读
    """
    x = x.float()
    y = y.float()
    
    # 计算x和y的平方范数（对角线元素）
    x_sqnorms = torch.diag(torch.mm(x, x.t()))
    y_sqnorms = torch.diag(torch.mm(y, y.t()))
    
    gamma = 1 / (2 * sigma ** 2)
    
    # 计算Kxx的均值
    k_xx = torch.exp(
        -gamma * (
            -2 * torch.mm(x, x.t())
            + x_sqnorms.unsqueeze(1)
            + x_sqnorms.unsqueeze(0)
        )
    ).mean()
    
    # 计算Kxy的均值
    k_xy = torch.exp(
        -gamma * (
            -2 * torch.mm(x, y.t())
            + x_sqnorms.unsqueeze(1)
            + y_sqnorms.unsqueeze(0)
        )
    ).mean()
    
    # 计算Kyy的均值
    k_yy = torch.exp(
        -gamma * (
            -2 * torch.mm(y, y.t())
            + y_sqnorms.unsqueeze(1)
            + y_sqnorms.unsqueeze(0)
        )
    ).mean()

    # 计算并返回缩放后的MMD值
    return scale * (k_xx + k_yy - 2 * k_xy)



def parse_args_paired_training(input_args=None):
    """
    Parses command-line arguments used for configuring an paired session (pix2pix-Turbo).
    This function sets up an argument parser to handle various training options.

    Returns:
    argparse.Namespace: The parsed command-line arguments.
   """
    parser = argparse.ArgumentParser()
    # args for the loss function
    parser.add_argument("--gan_disc_type", default="vagan_clip")
    parser.add_argument("--gan_loss_type", default="multilevel_sigmoid_s")
    parser.add_argument("--lambda_gan", default=0.5, type=float)
    parser.add_argument("--lambda_lpips", default=5, type=float)
    parser.add_argument("--lambda_l2", default=1.0, type=float)
    parser.add_argument("--lambda_clipsim", default=5.0, type=float)

    # dataset options
    parser.add_argument("--dataset_folder", required=True, type=str)
    parser.add_argument("--train_image_prep", default="resized_crop_512", type=str)
    parser.add_argument("--test_image_prep", default="resized_crop_512", type=str)

    # validation eval args
    parser.add_argument("--eval_freq", default=100, type=int)
    parser.add_argument("--track_val_fid", default=False, action="store_true")
    parser.add_argument("--num_samples_eval", type=int, default=100, help="Number of samples to use for all evaluation")

    parser.add_argument("--viz_freq", type=int, default=100, help="Frequency of visualizing the outputs.")
    parser.add_argument("--tracker_project_name", type=str, default="train_pix2pix_turbo", help="The name of the wandb project to log to.")

    # details about the model architecture
    parser.add_argument("--pretrained_model_name_or_path")
    parser.add_argument("--revision", type=str, default=None,)
    parser.add_argument("--variant", type=str, default=None,)
    parser.add_argument("--tokenizer_name", type=str, default=None)
    parser.add_argument("--lora_rank_unet", default=8, type=int)
    parser.add_argument("--lora_rank_vae", default=4, type=int)

    # training details
    parser.add_argument("--output_dir", required=True)
    parser.add_argument("--cache_dir", default=None,)
    parser.add_argument("--seed", type=int, default=None, help="A seed for reproducible training.")
    parser.add_argument("--resolution", type=int, default=512,)
    parser.add_argument("--train_batch_size", type=int, default=4, help="Batch size (per device) for the training dataloader.")
    parser.add_argument("--num_training_epochs", type=int, default=10)
    parser.add_argument("--max_train_steps", type=int, default=10_000,)
    parser.add_argument("--checkpointing_steps", type=int, default=500,)
    parser.add_argument("--gradient_accumulation_steps", type=int, default=1, help="Number of updates steps to accumulate before performing a backward/update pass.",)
    parser.add_argument("--gradient_checkpointing", action="store_true",)
    parser.add_argument("--learning_rate", type=float, default=5e-6)
    parser.add_argument("--lr_scheduler", type=str, default="constant",
        help=(
            'The scheduler type to use. Choose between ["linear", "cosine", "cosine_with_restarts", "polynomial",'
            ' "constant", "constant_with_warmup"]'
        ),
    )
    parser.add_argument("--lr_warmup_steps", type=int, default=500, help="Number of steps for the warmup in the lr scheduler.")
    parser.add_argument("--lr_num_cycles", type=int, default=1,
        help="Number of hard resets of the lr in cosine_with_restarts scheduler.",
    )
    parser.add_argument("--lr_power", type=float, default=1.0, help="Power factor of the polynomial scheduler.")

    parser.add_argument("--dataloader_num_workers", type=int, default=0,)
    parser.add_argument("--adam_beta1", type=float, default=0.9, help="The beta1 parameter for the Adam optimizer.")
    parser.add_argument("--adam_beta2", type=float, default=0.999, help="The beta2 parameter for the Adam optimizer.")
    parser.add_argument("--adam_weight_decay", type=float, default=1e-2, help="Weight decay to use.")
    parser.add_argument("--adam_epsilon", type=float, default=1e-08, help="Epsilon value for the Adam optimizer")
    parser.add_argument("--max_grad_norm", default=1.0, type=float, help="Max gradient norm.")
    parser.add_argument("--allow_tf32", action="store_true",
        help=(
            "Whether or not to allow TF32 on Ampere GPUs. Can be used to speed up training. For more information, see"
            " https://pytorch.org/docs/stable/notes/cuda.html#tensorfloat-32-tf32-on-ampere-devices"
        ),
    )
    parser.add_argument("--report_to", type=str, default="wandb",
        help=(
            'The integration to report the results and logs to. Supported platforms are `"tensorboard"`'
            ' (default), `"wandb"` and `"comet_ml"`. Use `"all"` to report to all integrations.'
        ),
    )
    parser.add_argument("--mixed_precision", type=str, default=None, choices=["no", "fp16", "bf16"],)
    parser.add_argument("--enable_xformers_memory_efficient_attention", action="store_true", help="Whether or not to use xformers.")
    parser.add_argument("--set_grads_to_none", action="store_true",)

    if input_args is not None:
        args = parser.parse_args(input_args)
    else:
        args = parser.parse_args()

    return args


def parse_args_unpaired_training():
    """
    Parses command-line arguments used for configuring an unpaired session (CycleGAN-Turbo).
    This function sets up an argument parser to handle various training options.

    Returns:
    argparse.Namespace: The parsed command-line arguments.
   """

    parser = argparse.ArgumentParser(description="Simple example of a ControlNet training script.")

    # fixed random seed
    parser.add_argument("--seed", type=int, default=42, help="A seed for reproducible training.")

    # args for the loss function
    parser.add_argument("--gan_disc_type", default="vagan_clip")
    parser.add_argument("--gan_loss_type", default="multilevel_sigmoid")
    parser.add_argument("--lambda_gan", default=0.5, type=float)
    parser.add_argument("--lambda_idt", default=1, type=float)
    parser.add_argument("--lambda_cycle", default=1, type=float)
    parser.add_argument("--lambda_cycle_lpips", default=10.0, type=float)
    parser.add_argument("--lambda_idt_lpips", default=1.0, type=float)
    parser.add_argument("--lambda_hist", default=1.0, type=float)
    parser.add_argument("--lambda_emd", default=100.0, type=float)
    parser.add_argument("--lambda_entropy", default=0.1, type=float)

    # args for dataset and dataloader options
    parser.add_argument("--dataset_folder", required=True, type=str)
    parser.add_argument("--train_img_prep", required=True)
    parser.add_argument("--val_img_prep", required=True)
    parser.add_argument("--dataloader_num_workers", type=int, default=0)
    parser.add_argument("--train_batch_size", type=int, default=4, help="Batch size (per device) for the training dataloader.")
    parser.add_argument("--max_train_epochs", type=int, default=100)
    parser.add_argument("--max_train_steps", type=int, default=None)
    parser.add_argument("--min_val", type=int, default=-1)
    parser.add_argument("--max_val", type=int, default=1)
    parser.add_argument("--bin_num", type=int, default=128)
    parser.add_argument("--kernel_width_ratio", type=float, default=2.5)

    # args for the model
    parser.add_argument("--pretrained_model_name_or_path", default="stabilityai/sd-turbo")
    parser.add_argument("--revision", default=None, type=str)
    parser.add_argument("--variant", default=None, type=str)
    parser.add_argument("--lora_rank_unet", default=128, type=int)
    parser.add_argument("--lora_rank_vae", default=4, type=int)

    # args for validation and logging
    parser.add_argument("--viz_freq", type=int, default=20)
    parser.add_argument("--output_dir", type=str, required=True)
    parser.add_argument("--report_to", type=str, default="wandb")
    parser.add_argument("--tracker_project_name", type=str, required=True)
    parser.add_argument("--validation_steps", type=int, default=500,)
    parser.add_argument("--validation_num_images", type=int, default=-1, help="Number of images to use for validation. -1 to use all images.")
    parser.add_argument("--checkpointing_steps", type=int, default=500)

    # args for the optimization options
    parser.add_argument("--learning_rate", type=float, default=5e-6,)
    parser.add_argument("--adam_beta1", type=float, default=0.9, help="The beta1 parameter for the Adam optimizer.")
    parser.add_argument("--adam_beta2", type=float, default=0.999, help="The beta2 parameter for the Adam optimizer.")
    parser.add_argument("--adam_weight_decay", type=float, default=1e-2, help="Weight decay to use.")
    parser.add_argument("--adam_epsilon", type=float, default=1e-08, help="Epsilon value for the Adam optimizer")
    parser.add_argument("--max_grad_norm", default=10.0, type=float, help="Max gradient norm.")
    parser.add_argument("--lr_scheduler", type=str, default="constant", help=(
        'The scheduler type to use. Choose between ["linear", "cosine", "cosine_with_restarts", "polynomial",'
        ' "constant", "constant_with_warmup"]'
        ),
    )
    parser.add_argument("--lr_warmup_steps", type=int, default=500, help="Number of steps for the warmup in the lr scheduler.")
    parser.add_argument("--lr_num_cycles", type=int, default=1, help="Number of hard resets of the lr in cosine_with_restarts scheduler.",)
    parser.add_argument("--lr_power", type=float, default=1.0, help="Power factor of the polynomial scheduler.")
    parser.add_argument("--gradient_accumulation_steps", type=int, default=1)

    # memory saving options
    parser.add_argument("--allow_tf32", action="store_true",
        help=(
            "Whether or not to allow TF32 on Ampere GPUs. Can be used to speed up training. For more information, see"
            " https://pytorch.org/docs/stable/notes/cuda.html#tensorfloat-32-tf32-on-ampere-devices"
        ),
    )
    parser.add_argument("--gradient_checkpointing", action="store_true",
        help="Whether or not to use gradient checkpointing to save memory at the expense of slower backward pass.")
    parser.add_argument("--enable_xformers_memory_efficient_attention", action="store_true", help="Whether or not to use xformers.")

    args = parser.parse_args()
    return args


def build_transform(image_prep):
    """
    Constructs a transformation pipeline based on the specified image preparation method.

    Parameters:
    - image_prep (str): A string describing the desired image preparation

    Returns:
    - torchvision.transforms.Compose: A composable sequence of transformations to be applied to images.
    """
    if image_prep == "resized_crop_512":
        T = transforms.Compose([
            transforms.Resize(512, interpolation=transforms.InterpolationMode.LANCZOS),
            transforms.CenterCrop(512),
        ])
    elif image_prep == "resize_286_randomcrop_256x256_hflip":
        T = transforms.Compose([
            transforms.Resize((286, 286), interpolation=Image.LANCZOS),
            transforms.RandomCrop((256, 256)),
            transforms.RandomHorizontalFlip(),
        ])
    elif image_prep in ["resize_256", "resize_256x256"]:
        T = transforms.Compose([
            transforms.Resize((256, 256), interpolation=Image.LANCZOS)
        ])
    elif image_prep in ["resize_512", "resize_512x512"]:
        T = transforms.Compose([
            transforms.Resize((512, 512), interpolation=Image.LANCZOS)
        ])
    elif image_prep == "no_resize":
        T = transforms.Lambda(lambda x: x)
    return T


class PairedDataset(torch.utils.data.Dataset):
    def __init__(self, dataset_folder, split, image_prep, tokenizer):
        """
        Itialize the paired dataset object for loading and transforming paired data samples
        from specified dataset folders.

        This constructor sets up the paths to input and output folders based on the specified 'split',
        loads the captions (or prompts) for the input images, and prepares the transformations and
        tokenizer to be applied on the data.

        Parameters:
        - dataset_folder (str): The root folder containing the dataset, expected to include
                                sub-folders for different splits (e.g., 'train_A', 'train_B').
        - split (str): The dataset split to use ('train' or 'test'), used to select the appropriate
                       sub-folders and caption files within the dataset folder.
        - image_prep (str): The image preprocessing transformation to apply to each image.
        - tokenizer: The tokenizer used for tokenizing the captions (or prompts).
        """
        super().__init__()
        if split == "train":
            self.input_folder = os.path.join(dataset_folder, "train_A")
            self.output_folder = os.path.join(dataset_folder, "train_B")
            captions = os.path.join(dataset_folder, "train_prompts.json")
        elif split == "test":
            self.input_folder = os.path.join(dataset_folder, "test_A")
            self.output_folder = os.path.join(dataset_folder, "test_B")
            captions = os.path.join(dataset_folder, "test_prompts.json")
        with open(captions, "r") as f:
            self.captions = json.load(f)
        self.img_names = list(self.captions.keys())
        self.T = build_transform(image_prep)
        self.tokenizer = tokenizer

    def __len__(self):
        """
        Returns:
        int: The total number of items in the dataset.
        """
        return len(self.captions)

    def __getitem__(self, idx):
        """
        Retrieves a dataset item given its index. Each item consists of an input image, 
        its corresponding output image, the captions associated with the input image, 
        and the tokenized form of this caption.

        This method performs the necessary preprocessing on both the input and output images, 
        including scaling and normalization, as well as tokenizing the caption using a provided tokenizer.

        Parameters:
        - idx (int): The index of the item to retrieve.

        Returns:
        dict: A dictionary containing the following key-value pairs:
            - "output_pixel_values": a tensor of the preprocessed output image with pixel values 
            scaled to [-1, 1].
            - "conditioning_pixel_values": a tensor of the preprocessed input image with pixel values 
            scaled to [0, 1].
            - "caption": the text caption.
            - "input_ids": a tensor of the tokenized caption.

        Note:
        The actual preprocessing steps (scaling and normalization) for images are defined externally 
        and passed to this class through the `image_prep` parameter during initialization. The 
        tokenization process relies on the `tokenizer` also provided at initialization, which 
        should be compatible with the models intended to be used with this dataset.
        """
        img_name = self.img_names[idx]
        input_img = Image.open(os.path.join(self.input_folder, img_name))
        output_img = Image.open(os.path.join(self.output_folder, img_name))
        caption = self.captions[img_name]

        # input images scaled to 0,1
        img_t = self.T(input_img)
        img_t = F.to_tensor(img_t)
        # output images scaled to -1,1
        output_t = self.T(output_img)
        output_t = F.to_tensor(output_t)
        output_t = F.normalize(output_t, mean=[0.5], std=[0.5])

        input_ids = self.tokenizer(
            caption, max_length=self.tokenizer.model_max_length,
            padding="max_length", truncation=True, return_tensors="pt"
        ).input_ids

        return {
            "output_pixel_values": output_t,
            "conditioning_pixel_values": img_t,
            "caption": caption,
            "input_ids": input_ids,
        }


class UnpairedDataset(torch.utils.data.Dataset):
    def __init__(self, dataset_folder, split, image_prep, tokenizer):
        """
        A dataset class for loading unpaired data samples from two distinct domains (source and target),
        typically used in unsupervised learning tasks like image-to-image translation.

        The class supports loading images from specified dataset folders, applying predefined image
        preprocessing transformations, and utilizing fixed textual prompts (captions) for each domain,
        tokenized using a provided tokenizer.

        Parameters:
        - dataset_folder (str): Base directory of the dataset containing subdirectories (train_A, train_B, test_A, test_B)
        - split (str): Indicates the dataset split to use. Expected values are 'train' or 'test'.
        - image_prep (str): he image preprocessing transformation to apply to each image.
        - tokenizer: The tokenizer used for tokenizing the captions (or prompts).
        """
        super().__init__()
        self.split = split
        if split == "train":
            self.source_folder = os.path.join(dataset_folder, "train_A")
            self.target_folder = os.path.join(dataset_folder, "train_B")
        elif split == "test":
            self.source_folder = os.path.join(dataset_folder, "test_A")
            self.target_folder = os.path.join(dataset_folder, "test_B")
        self.tokenizer = tokenizer
        with open(os.path.join(dataset_folder, "fixed_prompt_a.txt"), "r") as f:
            self.fixed_caption_src = f.read().strip()
            self.input_ids_src = self.tokenizer(
                self.fixed_caption_src, max_length=self.tokenizer.model_max_length,
                padding="max_length", truncation=True, return_tensors="pt"
            ).input_ids

        with open(os.path.join(dataset_folder, "fixed_prompt_b.txt"), "r") as f:
            self.fixed_caption_tgt = f.read().strip()
            self.input_ids_tgt = self.tokenizer(
                self.fixed_caption_tgt, max_length=self.tokenizer.model_max_length,
                padding="max_length", truncation=True, return_tensors="pt"
            ).input_ids
        # find all images in the source and target folders with all IMG extensions
        self.l_imgs_src = []
        for ext in ["*.jpg", "*.jpeg", "*.png", "*.bmp", "*.gif"]:
            self.l_imgs_src.extend(glob(os.path.join(self.source_folder, ext)))
        self.l_imgs_tgt = []
        for ext in ["*.jpg", "*.jpeg", "*.png", "*.bmp", "*.gif"]:
            self.l_imgs_tgt.extend(glob(os.path.join(self.target_folder, ext)))
        self.T = build_transform(image_prep)

    def __len__(self):
        """
        Returns:
        int: The total number of items in the dataset.
        """
        return len(self.l_imgs_src) + len(self.l_imgs_tgt)

    def __getitem__(self, index):
        """
        Fetches a pair of unaligned images from the source and target domains along with their 
        corresponding tokenized captions.

        For the source domain, if the requested index is within the range of available images,
        the specific image at that index is chosen. If the index exceeds the number of source
        images, a random source image is selected. For the target domain,
        an image is always randomly selected, irrespective of the index, to maintain the 
        unpaired nature of the dataset.

        Both images are preprocessed according to the specified image transformation `T`, and normalized.
        The fixed captions for both domains
        are included along with their tokenized forms.

        Parameters:
        - index (int): The index of the source image to retrieve.

        Returns:
        dict: A dictionary containing processed data for a single training example, with the following keys:
            - "pixel_values_src": The processed source image
            - "pixel_values_tgt": The processed target image
            - "caption_src": The fixed caption of the source domain.
            - "caption_tgt": The fixed caption of the target domain.
            - "input_ids_src": The source domain's fixed caption tokenized.
            - "input_ids_tgt": The target domain's fixed caption tokenized.
            - "condition_src": The condition for the source image
            - "condition_tgt": The condition for the target image
        """
        if index < len(self.l_imgs_src):
            img_path_src = self.l_imgs_src[index]
        else:
        # 训练时随机选择，测试时按索引取模（保证固定）
            if self.split == "train":
                img_path_src = random.choice(self.l_imgs_src)
            else:
                img_path_src = self.l_imgs_src[index % len(self.l_imgs_src)]
    
        # 目标域图像选择（区分训练/测试）
        if self.split == "train":
        # 训练时随机选择，增强泛化性
            img_path_tgt = random.choice(self.l_imgs_tgt)
        else:
            # 测试时按索引取模，保证每次选择固定
            img_path_tgt = self.l_imgs_tgt[index % len(self.l_imgs_tgt)]
            
        img_path_tgt = random.choice(self.l_imgs_tgt)
        img_pil_src = Image.open(img_path_src).convert("RGB")
        img_pil_tgt = Image.open(img_path_tgt).convert("RGB")
        
        img_t_src = F.to_tensor(self.T(img_pil_src))
        img_t_tgt = F.to_tensor(self.T(img_pil_tgt))

        img_t_src = F.normalize(img_t_src, mean=[0.5], std=[0.5])
        img_t_tgt = F.normalize(img_t_tgt, mean=[0.5], std=[0.5])


        return {
            "pixel_values_src": img_t_src,
            "pixel_values_tgt": img_t_tgt,
            "caption_src": self.fixed_caption_src,
            "caption_tgt": self.fixed_caption_tgt,
            "input_ids_src": self.input_ids_src,
            "input_ids_tgt": self.input_ids_tgt

        }
