import os
import shutil
import numpy as np
import torch
from torch import Tensor
from torchmetrics import StructuralSimilarityIndexMeasure
from PIL import Image
import torchvision.transforms as transforms
# import cleanfid
from cleanfid.fid import build_feature_extractor, get_folder_features, frechet_distance
from typing import List, Dict, Tuple, Optional, Union
from options.test_options import TestOptions
from data import create_dataset
from models import create_model
from util.visualizer import save_images
from util import html
import util.util as util
from data.image_folder import make_dataset  # 用于加载目标域参考图像


def compute_cmmd(x: Union[np.ndarray, Tensor], y: Union[np.ndarray, Tensor], 
                sigma: float = 10.0, scale: float = 1000.0) -> float:
    """
    计算跨模态最大均值差异（基于PyTorch的高效实现）
    
    Args:
        x: 第一组特征数据
        y: 第二组特征数据
        sigma: 高斯核带宽参数
        scale: 结果缩放因子
        
    Returns:
        缩放后的MMD值
    """
    # 转换为PyTorch张量并确保浮点类型
    if isinstance(x, np.ndarray):
        x = torch.from_numpy(x).float()
    if isinstance(y, np.ndarray):
        y = torch.from_numpy(y).float()
    
    # 展平特征（确保形状为[batch_size, feature_dim]）
    x = x.reshape(x.shape[0], -1)
    y = y.reshape(y.shape[0], -1)
    
    # 计算x和y的平方范数（对角线元素）
    x_sqnorms = torch.diag(torch.mm(x, x.t()))
    y_sqnorms = torch.diag(torch.mm(y, y.t()))
    
    gamma = 1.0 / (2 * sigma **2)
    
    # 计算核函数值及其均值
    k_xx = torch.exp(-gamma * (-2 * torch.mm(x, x.t()) + x_sqnorms.unsqueeze(1) + x_sqnorms.unsqueeze(0))).mean()
    k_xy = torch.exp(-gamma * (-2 * torch.mm(x, y.t()) + x_sqnorms.unsqueeze(1) + y_sqnorms.unsqueeze(0))).mean()
    k_yy = torch.exp(-gamma * (-2 * torch.mm(y, y.t()) + y_sqnorms.unsqueeze(1) + y_sqnorms.unsqueeze(0))).mean()
    
    return scale * (k_xx + k_yy - 2 * k_xy).item()


def calculate_metrics(source_images: List[np.ndarray], fake_images: List[np.ndarray],
                     real_features: np.ndarray, feature_extractor,
                     ssim_metric: StructuralSimilarityIndexMeasure,
                     device: str) -> Tuple[float, float, float]:
    """
    计算一组生成图像的SSIM、FID和CMMD指标
    
    Args:
        source_images: 源域图像列表（用于SSIM计算）
        fake_images: 生成图像列表
        real_features: 目标域真实图像的特征（用于FID和CMMD）
        feature_extractor: 特征提取器
        ssim_metric: SSIM计算对象
        device: 计算设备（cuda或cpu）
        
    Returns:
        平均SSIM、FID距离和CMMD值
    """
    # 计算SSIM（源域图像与生成图像对比）
    ssim_scores = []
    for source, fake in zip(source_images, fake_images):
        source_tensor = torch.from_numpy(source).to(device)
        fake_tensor = torch.from_numpy(fake).to(device)
        ssim_val = ssim_metric(source_tensor, fake_tensor).item()
        ssim_scores.append(ssim_val)
    avg_ssim = np.mean(ssim_scores)
    
    # 计算FID和生成特征
    ref_mu = np.mean(real_features, axis=0)
    ref_sigma = np.cov(real_features, rowvar=False)
    
    # 为FID计算生成临时目录
    temp_gen_dir = "temp_fid_generated"
    os.makedirs(temp_gen_dir, exist_ok=True)
    
    # 保存生成图像用于FID计算
    for i, img in enumerate(fake_images):
        try:
            img_np = img.squeeze().transpose(1, 2, 0)
            img_np = ((img_np + 1) / 2 * 255).astype(np.uint8)
            img_pil = Image.fromarray(img_np)
            img_path = os.path.join(temp_gen_dir, f"img_{i:04d}.png")
            if not os.path.exists(img_path):
                img_pil.save(img_path)
        except Exception as e:
            print(f"保存生成图像时出错: {str(e)}")
    
    # 提取生成图像特征
    gen_features = get_folder_features(
        temp_gen_dir,
        model=feature_extractor,
        num_workers=0,
        num=None,
        shuffle=False,
        seed=0,
        batch_size=8,
        device=torch.device(device),
        mode="clean",
        custom_fn_resize=None,
        description="计算生成图像特征",
        verbose=True,
        custom_image_tranform=None
    )
    
    gen_mu = np.mean(gen_features, axis=0)
    gen_sigma = np.cov(gen_features, rowvar=False)
    fid = frechet_distance(ref_mu, ref_sigma, gen_mu, gen_sigma)
    
    # 清理临时目录
    if os.path.exists(temp_gen_dir):
        shutil.rmtree(temp_gen_dir)
    
    # 计算CMMD - 使用FID的特征并进行样本对齐
    min_samples = min(len(real_features), len(gen_features))
    real_feat_aligned = real_features[:min_samples]
    gen_feat_aligned = gen_features[:min_samples]
    
    real_tensor = torch.from_numpy(real_feat_aligned).float().to(device)
    gen_tensor = torch.from_numpy(gen_feat_aligned).float().to(device)
    
    cmmd = compute_cmmd(real_tensor, gen_tensor)
    
    return avg_ssim, fid, cmmd


def load_target_domain_images(target_dir: str, transform, max_num: Optional[int] = None) -> List[np.ndarray]:
    """
    加载目标域图像并应用转换
    
    Args:
        target_dir: 目标域图像目录（testB）
        transform: 图像转换函数
        max_num: 最大加载数量
        
    Returns:
        处理后的目标域图像列表
    """
    target_images = []
    image_paths = make_dataset(target_dir)  # 使用数据加载工具获取图像路径
    if max_num is not None:
        image_paths = image_paths[:max_num]
    
    for path in image_paths:
        try:
            img = Image.open(path).convert('RGB')
            img_tensor = transform(img)  # 应用转换（包括 resize 到 256x256）
            target_images.append(img_tensor.unsqueeze(0).numpy())  # 保持与生成图像相同的形状
        except Exception as e:
            print(f"加载目标域图像 {path} 时出错: {str(e)}")
    
    return target_images


if __name__ == '__main__':
    # 解析配置
    opt = TestOptions().parse()
    
    # 硬编码测试参数
    opt.num_threads = 0    # 测试代码仅支持单线程
    opt.batch_size = 1     # 测试代码仅支持batch_size=1
    opt.serial_batches = True  # 禁用数据打乱
    opt.no_flip = True     # 不翻转图像
    opt.display_id = -1    # 不使用visdom显示
    target_domain_dir = os.path.join(opt.dataroot, "testB")  # 目标域图像目录

    # 定义图像转换（256x256）
    def get_transform():
        transform_list = [
            transforms.Resize((256, 256), transforms.InterpolationMode.BILINEAR),
            transforms.ToTensor(),
            transforms.Normalize((0.5, 0.5, 0.5), (0.5, 0.5, 0.5))
        ]
        return transforms.Compose(transform_list)
    
    transform = get_transform()
    
    # 创建数据集和模型
    try:
        dataset = create_dataset(opt)
        model = create_model(opt)
        
        # 初始化模型
        model.setup(opt)
        model.parallelize()
        if opt.eval:
            model.eval()
        
    except Exception as e:
        print(f"初始化数据集或模型失败: {str(e)}")
        exit(1)
    
    # 创建结果网页目录
    web_dir = os.path.join(opt.results_dir, opt.name, f'{opt.phase}_{opt.epoch}')
    print(f'创建网页目录: {web_dir}')
    os.makedirs(web_dir, exist_ok=True)
    webpage = html.HTML(web_dir, f'Experiment = {opt.name}, Phase = {opt.phase}, Epoch = {opt.epoch}')
    
    # 初始化存储结构
    source_images = []  # 源域图像（用于SSIM计算）
    fake_images = []    # 生成图像
    device = "cuda" if torch.cuda.is_available() else "cpu"
    ssim_metric = StructuralSimilarityIndexMeasure(data_range=1.0).to(device)
    
    # 处理图像并生成结果
    try:
        for i, data in enumerate(dataset):
            if i >= opt.num_test:
                break
                
            model.set_input(data)
            if i == 0:
                model.data_dependent_initialize()
                model.setup(opt)
                model.parallelize()
                if opt.eval:
                    model.eval()
                    
            model.test()
            visuals = model.get_current_visuals()
            img_path = model.get_image_paths()
            
            # 收集源域和生成图像
            if 'real_A' in visuals:
                source_images.append(visuals['real_A'].cpu().numpy())
            if 'fake_B' in visuals:
                fake_images.append(visuals['fake_B'].cpu().numpy())
            
            # 定期保存到HTML
            if i % 5 == 0:
                print(f'处理第{i:04d}张图像... {img_path}')
            save_images(webpage, visuals, img_path, width=opt.display_winsize)
        
        webpage.save()
        print("HTML结果已保存")
        
    except Exception as e:
        print(f"图像处理过程出错: {str(e)}")
        exit(1)
    
    # 加载目标域参考图像（testB）
    print(f"从 {target_domain_dir} 加载目标域参考图像...")
    target_images = load_target_domain_images(
        target_domain_dir, 
        transform, 
        max_num=opt.num_test  # 与测试样本数保持一致
    )
    if not target_images:
        print("错误: 未能加载任何目标域参考图像，无法计算FID和CMMD")
        exit(1)
    print(f"成功加载 {len(target_images)} 张目标域参考图像")
    
    # 准备特征提取器并提取目标域特征
    try:
        feature_extractor = build_feature_extractor("clean", device=device, use_dataparallel=False)
        
        # 保存目标域参考图像用于特征提取
        ref_dir = os.path.join(web_dir, "target_reference_images")
        os.makedirs(ref_dir, exist_ok=True)
        
        for i, img in enumerate(target_images):
            try:
                img_np = img.squeeze().transpose(1, 2, 0)
                img_np = ((img_np + 1) / 2 * 255).astype(np.uint8)
                img_pil = Image.fromarray(img_np)
                img_path = os.path.join(ref_dir, f"target_ref_{i:04d}.png")
                if not os.path.exists(img_path):
                    img_pil.save(img_path)
            except Exception as e:
                print(f"保存目标域参考图像时出错: {str(e)}")
        
        # 提取目标域参考图像特征
        target_features = get_folder_features(
            ref_dir,
            model=feature_extractor,
            num_workers=0,
            num=None,
            shuffle=False,
            seed=0,
            batch_size=8,
            device=torch.device(device),
            mode="clean",
            custom_fn_resize=None,
            description="计算目标域参考图像特征",
            verbose=True,
            custom_image_tranform=None
        )
        
        # 保存CMMD参考特征
        if hasattr(opt, 'direction') and opt.direction == 'AtoB':
            np.save(os.path.join(web_dir, "cmmd_target_a2b.npy"), target_features)
        else:
            np.save(os.path.join(web_dir, "cmmd_target_b2a.npy"), target_features)
            
    except Exception as e:
        print(f"特征提取器初始化失败: {str(e)}")
        exit(1)
    
    # 计算并保存指标
    if fake_images and len(source_images) == len(fake_images):
        try:
            avg_ssim, fid, cmmd = calculate_metrics(
                source_images, fake_images, target_features,
                feature_extractor, ssim_metric, device
            )
            
            print(f"\n===== 评价指标 =====")
            print(f"平均SSIM (源域 vs 生成): {avg_ssim:.4f}")
            print(f"FID距离 (目标域 vs 生成): {fid:.4f}")
            print(f"CMMD值 (目标域 vs 生成): {cmmd:.4f}")
            
            # 保存指标结果
            with open(os.path.join(web_dir, 'metrics.txt'), 'w', encoding='utf-8') as f:
                f.write(f"平均SSIM (源域 vs 生成): {avg_ssim:.4f}\n")
                f.write(f"FID距离 (目标域 vs 生成): {fid:.4f}\n")
                f.write(f"CMMD值 (目标域 vs 生成): {cmmd:.4f}\n")
                
        except Exception as e:
            print(f"计算指标时出错: {str(e)}")
    else:
        print("\n警告: 生成图像与源域图像数量不匹配，无法计算指标")