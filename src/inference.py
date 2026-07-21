import os
import argparse
from PIL import Image
import torch
from torchvision import transforms
from cyclegan_turbo import CycleGAN_Turbo
from my_utils.training_utils import build_transform
from my_utils.condition_encoder import HistogramLikeEncoder  


if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument('--input_image', type=str, required=True, help='path to the input image')
    parser.add_argument('--prompt', type=str, required=False, help='the prompt to be used. It is required when loading a custom model_path.')
    parser.add_argument('--model_name', type=str, default=None, help='name of the pretrained model to be used')
    parser.add_argument('--model_path', type=str, default=None, help='path to a local model state dict to be used')
    parser.add_argument('--output_dir', type=str, default='output', help='the directory to save the output')
    parser.add_argument('--image_prep', type=str, default='resize_512x512', help='the image preparation method')
    parser.add_argument('--direction', type=str, default=None, help='the direction of translation. None for pretrained models, a2b or b2a for custom paths.')
    parser.add_argument('--use_fp16', action='store_true', help='Use Float16 precision for faster inference')
    args = parser.parse_args()

    # only one of model_name and model_path should be provided
    if (args.model_name is None) == (args.model_path is None):
        raise ValueError('Either model_name or model_path should be provided')

    if args.model_path is not None and args.prompt is None:
        raise ValueError('prompt is required when loading a custom model_path.')

    if args.model_name is not None:
        assert args.prompt is None, 'prompt is not required when loading a pretrained model.'
        assert args.direction is None, 'direction is not required when loading a pretrained model.'

    # initialize the model
    model = CycleGAN_Turbo(pretrained_name=args.model_name, pretrained_path=args.model_path)
    model.eval()
    model.unet.enable_xformers_memory_efficient_attention()
    if args.use_fp16:
        model.half()

    hist_encoder = HistogramLikeEncoder(output_dim=1024).cuda()
    if args.model_path is not None:
        ckpt = torch.load(args.model_path, map_location="cuda")
        load_res = hist_encoder.load_state_dict(ckpt["sd_hist_encoder"], strict=True)

    if args.use_fp16:
        hist_encoder.half()

    T_val = build_transform(args.image_prep)

    input_image = Image.open(args.input_image).convert('RGB')
    # translate the image
    with torch.no_grad():
        input_img = T_val(input_image)
        x_t = transforms.ToTensor()(input_img)
        x_t = transforms.Normalize([0.5], [0.5])(x_t).unsqueeze(0).cuda()
        if args.use_fp16:
            x_t = x_t.half()

        condition = hist_encoder(x_t)

        output = model(x_t, direction=args.direction, caption=args.prompt, condition=condition)

    bname = os.path.basename(args.input_image)
    os.makedirs(args.output_dir, exist_ok=True)

    output_pil_256 = transforms.ToPILImage()(output[0].cpu() * 0.5 + 0.5)
    output_pil_256.save(os.path.join(args.output_dir, bname))
