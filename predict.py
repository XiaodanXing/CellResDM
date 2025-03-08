#!/usr/bin/env python

import argparse
from pathlib import Path
from omegaconf import OmegaConf
from sampler import ResShiftSampler
from utils.util_opts import str2bool
from datapipe.datasets import create_dataset

def get_parser(**parser_kwargs):
    parser = argparse.ArgumentParser(**parser_kwargs)
    parser.add_argument("-i", "--in_path", type=str, default="/media/NAS06/chunling/virtual", help="Input path.")
    parser.add_argument("-o", "--out_path", type=str, default="/media/NAS06/chunling/virtual", help="Output path.")
    parser.add_argument("--mask_path", type=str, default="", help="Mask path for inpainting.")
    parser.add_argument("--scale", type=int, default=1, help="Scale factor for SR. In image translation tasks, this is always set to 1.")
    parser.add_argument("--seed", type=int, default=12345, help="Random seed.")
    parser.add_argument("--bs", type=int, default=1, help="Batch size.")
    parser.add_argument("--chop_size", type=int, default=512, choices=[512, 256, 64], help="Chopping forward. Used when the input image is large.")
    parser.add_argument("--chop_stride", type=int, default=-1, help="Chopping stride.")
    parser.add_argument("--task", type=str, default="custom_noae", choices=['custom', 'custom_noae', 'custom_3channel', 'custom_wmask'], help="Task type")
    args = parser.parse_args()
    return args

def get_configs(args):
    ckpt_dir = Path('./weights')
    if not ckpt_dir.exists():
        ckpt_dir.mkdir()

    if args.task == 'custom':
        configs = OmegaConf.load('./configs/custom_5channelae.yaml')
        ckpt_path = Path('./save_dir/5channelAE-input-6-6-6-7-8/ckpts/model_400000.pth')
        vqgan_path = Path('/home/xiaodan/PycharmProjects/IPMI2023/diffusion_model/logs/5channel_nomask/checkpoints/last.ckpt')
    elif args.task == 'custom_noae':
        configs = OmegaConf.load('./configs/custom_noae.yaml')
        ckpt_path = Path('./save_dir_noae/2024-09-26-13-12/ckpts/model_60000.pth')
        vqgan_path = ckpt_dir / 'autoencoder_vq_f4.pth'
    elif args.task == 'custom_3channel':
        configs = OmegaConf.load('./configs/custom.yaml')
        ckpt_path = Path('./save_dir/ch-1-2-5-naturalAE-input-6-7-8/ckpts/model_400000.pth')
        vqgan_path = ckpt_dir / 'autoencoder_vq_f4.pth'
    elif args.task == 'custom_wmask':
        configs = OmegaConf.load('./configs/custom_wmask.yaml')
        ckpt_path = Path('./save_dir/7channelwmask-input-6-6-6-6-6-7-8/ckpts/model_340000.pth')
        vqgan_path = Path('/home/xiaodan/PycharmProjects/IPMI2023/diffusion_model/logs/5channel_mask/checkpoints/last.ckpt')
    elif args.task == 'noae_wmask':
        configs = OmegaConf.load('./configs/custom_noae_wmask.yaml')
        ckpt_path = Path('./save_dir_noae/noAE_wmask/ckpts/model_340000.pth')
        vqgan_path = Path('/home/xiaodan/PycharmProjects/IPMI2023/diffusion_model/logs/5channel_mask/checkpoints/last.ckpt')
    else:
        raise TypeError(f"Unexpected task type: {args.task}!")

    # Prepare the checkpoint
    configs.model.ckpt_path = str(ckpt_path)
    configs.diffusion.params.sf = args.scale
    configs.autoencoder.ckpt_path = str(vqgan_path)

    # Save folder
    if not Path(args.out_path).exists():
        Path(args.out_path).mkdir(parents=True)

    if args.chop_stride < 0:
        if args.chop_size == 512:
            chop_stride = (512 - 64) * (4 // args.scale)
        elif args.chop_size == 256:
            chop_stride = (256 - 32) * (4 // args.scale)
        elif args.chop_size == 64:
            chop_stride = (64 - 16) * (4 // args.scale)
        else:
            raise ValueError("Chop size must be in [512, 256]")
    else:
        chop_stride = args.chop_stride * (4 // args.scale)
    args.chop_size *= (4 // args.scale)
    print(f"Chopping size/stride: {args.chop_size}/{chop_stride}")

    return configs, chop_stride


def get_dataset(in_path):
    data_config = {
        'type': 'custom',
        'params': {
            'dir_paths': str(in_path),
            'purpose': 'test',
            'image_size': 512,
            'cell_type': 'both',
            'transform_type': 'default',
            'transform_kwargs': {
                'mean': 0.5,
                'std': 0.5,
            },
            'need_path': True,
            'recursive': True,
            'length': None,
        }
    }
    dataset = create_dataset(data_config)
    return dataset

def main():
    args = get_parser()
    configs, chop_stride = get_configs(args)

    resshift_sampler = ResShiftSampler(
        configs,
        sf=args.scale,
        chop_size=args.chop_size,
        chop_stride=chop_stride,
        chop_bs=1,
        use_amp=True,
        seed=args.seed,
        padding_offset=configs.model.params.get('lq_size', 64),
    )

    # Setting mask path for inpainting
    if args.task.startswith('inpaint'):
        assert args.mask_path, 'Please input the mask path for inpainting!'
        mask_path = args.mask_path
    else:
        mask_path = None

    dataset = get_dataset(args.in_path) if Path(args.in_path).is_dir() else None

    resshift_sampler.inference(
        args.in_path,
        args.out_path,
        mask_path=mask_path,
        bs=args.bs,
        noise_repeat=False,
        dataset=dataset
    )

if __name__ == '__main__':
    main()
