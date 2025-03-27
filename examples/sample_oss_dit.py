import torch
import torch.distributed as dist
from models import DiT_models
from download import find_model

from diffusion import create_diffusion
from diffusers.models import AutoencoderKL
from tqdm import tqdm
import os
from PIL import Image
import numpy as np
import math
import argparse
import time
import torch.backends.cudnn as cudnn

from PIL import ImageDraw, Image, ImageFont


import sys
sys.path.append('../OSS')
from OSS.OSS import search_OSS, infer_OSS, cal_medium, search_OSS_batch
from OSS.model_wrap import _WrappedModel_DiT

def create_npz_from_sample_folder(sample_dir, num=50_000):
    """
    Builds a single .npz file from a folder of .png samples.
    """
    samples = []
    for i in tqdm(range(num), desc="Building .npz file from samples"):
        sample_pil = Image.open(f"{sample_dir}/{i:06d}.png")
        sample_np = np.asarray(sample_pil).astype(np.uint8)
        samples.append(sample_np)
    samples = np.stack(samples)
    assert samples.shape == (num, samples.shape[1], samples.shape[2], 3)
    npz_path = f"{sample_dir}.npz"
    np.savez(npz_path, arr_0=samples)
    print(f"Saved .npz file to {npz_path} [shape={samples.shape}].")
    return npz_path


def main(args):
    """
    Run sampling.
    """
    torch.backends.cuda.matmul.allow_tf32 = args.tf32  # True: fast but may lead to some small numerical differences
    assert torch.cuda.is_available(), "Sampling with DDP requires at least one GPU. sample.py supports CPU-only usage"
    torch.set_grad_enabled(False)

    # Setup DDP:
    dist.init_process_group("nccl")
    rank = dist.get_rank()
    device = rank % torch.cuda.device_count()
    seed = args.global_seed * dist.get_world_size() + rank
    torch.manual_seed(seed)
    torch.cuda.set_device(device)
    np.random.seed(seed)
    cudnn.benchmark = True
    print(f"Starting rank={rank}, seed={seed}, world_size={dist.get_world_size()}.")

    if args.ckpt is None:
        assert args.model == "DiT-XL/2", "Only DiT-XL/2 models are available for auto-download."
        assert args.image_size in [256, 512]
        assert args.num_classes == 1000

    # Load model:
    latent_size = args.image_size // 8
    model = DiT_models[args.model](
        input_size=latent_size,
        num_classes=args.num_classes
    ).to(device)
    # Auto-download a pre-trained model or load a custom DiT checkpoint from train.py:
    ckpt_path = args.ckpt or f"DiT-XL-2-{args.image_size}x{args.image_size}.pt"
    state_dict = torch.load(ckpt_path, map_location=lambda storage, loc: storage)
    model.load_state_dict(state_dict)
    model.eval()  # important!

    assert args.search_batch < 1000

    labels_gen = torch.tensor([0]*args.search_batch)
    y = torch.Tensor(labels_gen).long().cuda()
    if args.cfg_scale > 1.0:
        y_null = torch.tensor([1000] * args.search_batch, device=device)
        model_kwargs = dict( cfg_scale=args.cfg_scale)
        sample_fn = model.forward_with_cfg
    else:
        model_kwargs = dict()
        sample_fn = model.forward
        
    

    diffusion = create_diffusion(str(args.teacher_steps))
    z = torch.randn(args.search_batch, model.in_channels, latent_size, latent_size, device=device)
    wrap_model = _WrappedModel_DiT(sample_fn, diffusion, device, y_null)

    if not args.search_each:
        if args.batch_search:
            oss_steps = search_OSS_batch(wrap_model, z, args.search_batch, y, device, teacher_steps=args.teacher_steps, student_steps=args.student_steps, model_kwargs=model_kwargs)
        else:
            oss_steps = search_OSS(wrap_model, z, args.search_batch, y, device, teacher_steps=args.teacher_steps, student_steps=args.student_steps, model_kwargs=model_kwargs)

        oss_steps_tensor = torch.tensor(oss_steps, device=device)
        oss_steps_all_list = [torch.zeros_like(oss_steps_tensor) for _ in range(dist.get_world_size())] 

        torch.distributed.barrier()
        torch.distributed.all_gather(oss_steps_all_list, oss_steps_tensor)
        oss_steps_all_list = torch.cat(oss_steps_all_list, dim=0).tolist()
        oss_steps = cal_medium(oss_steps_all_list)
        print(f"OSS steps: {oss_steps}")



    vae = AutoencoderKL.from_pretrained(f"stabilityai/sd-vae-ft-{args.vae}").to(device)
    assert args.cfg_scale >= 1.0, "In almost all cases, cfg_scale be >= 1.0"
    using_cfg = args.cfg_scale > 1.0

    folder_name = args.folder_name                  
    sample_folder_dir = f"{args.sample_dir}/{folder_name}"

    if rank == 0:
        os.makedirs(sample_folder_dir, exist_ok=True)
        print(f"Saving .png samples at {sample_folder_dir}")
    dist.barrier()



    world_size = dist.get_world_size()
    local_rank = dist.get_rank()
    
    # Figure out how many samples we need to generate on each GPU and how many iterations we need to run:
    batch_size = 1 #args.num_fid_samples // args.num_classes
    # assert args.num_fid_samples % args.num_classes == 0, "total_samples must be divisible by num_class"
    assert args.num_classes % world_size == 0, "num_class must be divisible by world_size"
    iterations = int(args.num_classes // world_size)
    pbar = range(iterations)
    pbar = tqdm(pbar) if rank == 0 else pbar


    max_amp = [1.311  ,1.287 ,1.268, 1.251, 1.235, 1.22 , 1.206, 1.192, 1.18  ,1.169 ,1.158 ,1.148,
                1.139 ,1.13  ,1.122, 1.114, 1.107, 1.101, 1.094, 1.089, 1.084 ,1.08  ,1.076 ,1.072,
                1.068 ,1.065 ,1.063, 1.061, 1.059, 1.057, 1.056, 1.055, 1.054 ,1.054 ,1.053 ,1.054,
                1.054 ,1.055 ,1.055, 1.057, 1.058, 1.059, 1.061, 1.062, 1.064 ,1.067 ,1.069 ,1.071,
                1.074 ,1.077 ,1.079, 1.082, 1.086, 1.089, 1.092, 1.096, 1.099 ,1.103 ,1.107 ,1.11,
                1.114 ,1.118 ,1.123, 1.127, 1.132, 1.136, 1.141, 1.146, 1.151 ,1.156 ,1.16  ,1.165,
                1.17  ,1.176 ,1.182, 1.188, 1.193, 1.199, 1.204, 1.21 , 1.215 ,1.221 ,1.227 ,1.232,
                1.238 ,1.244 ,1.25 , 1.256, 1.262, 1.268, 1.274, 1.28 , 1.286 ,1.292 ,1.298 ,1.304,
                1.31  ,1.316 ,1.323, 1.328, 1.334, 1.34 , 1.346, 1.352, 1.358 ,1.364 ,1.37  ,1.376,
                1.382 ,1.388 ,1.393, 1.399, 1.405, 1.411, 1.416, 1.422, 1.427 ,1.432 ,1.438 ,1.443,
                1.448 ,1.454 ,1.459, 1.464, 1.469, 1.475, 1.48 , 1.484, 1.489 ,1.494 ,1.498 ,1.503,
                1.507 ,1.511 ,1.516, 1.52 , 1.524, 1.528, 1.532, 1.536, 1.54  ,1.544 ,1.547 ,1.551,
                1.554 ,1.558 ,1.561, 1.564, 1.567, 1.57 , 1.573, 1.576, 1.579 ,1.582 ,1.585 ,1.587,
                1.59  ,1.592 ,1.595, 1.597, 1.599, 1.601, 1.603, 1.605, 1.607 ,1.609 ,1.611 ,1.613,
                1.615 ,1.616 ,1.618, 1.619, 1.621, 1.622, 1.624, 1.625, 1.626 ,1.628 ,1.629 ,1.63,
                1.631 ,1.632 ,1.633, 1.634, 1.635, 1.636, 1.637, 1.637, 1.638 ,1.639 ,1.64  ,1.64,
                1.641 ,1.642 ,1.642, 1.643, 1.643, 1.644, 1.644, 1.645,
            ]

    min_amp = [-1.309  ,-1.285 ,-1.266 ,-1.248 ,-1.232 ,-1.217 ,-1.202 ,-1.189 ,-1.176 ,-1.163,
                -1.152 ,-1.141 ,-1.131 ,-1.122 ,-1.113 ,-1.104 ,-1.096 ,-1.089 ,-1.082 ,-1.075,
                -1.069 ,-1.063 ,-1.058 ,-1.053 ,-1.049 ,-1.044 ,-1.041 ,-1.037 ,-1.034 ,-1.032,
                -1.03  ,-1.028 ,-1.027 ,-1.026 ,-1.025 ,-1.024 ,-1.024 ,-1.024 ,-1.024 ,-1.024,
                -1.025 ,-1.026 ,-1.027 ,-1.029 ,-1.03  ,-1.032 ,-1.034 ,-1.036 ,-1.038 ,-1.041,
                -1.044 ,-1.047 ,-1.05  ,-1.053 ,-1.056 ,-1.06  ,-1.063 ,-1.067 ,-1.071 ,-1.076,
                -1.08  ,-1.084 ,-1.089 ,-1.093 ,-1.098 ,-1.103 ,-1.108 ,-1.113 ,-1.118 ,-1.123,
                -1.128 ,-1.134 ,-1.139 ,-1.144 ,-1.15  ,-1.157 ,-1.163 ,-1.169 ,-1.175 ,-1.181,
                -1.187 ,-1.193 ,-1.199 ,-1.205 ,-1.211 ,-1.217 ,-1.223 ,-1.23  ,-1.236 ,-1.242,
                -1.249 ,-1.255 ,-1.262 ,-1.268 ,-1.275 ,-1.281 ,-1.287 ,-1.294 ,-1.301 ,-1.307,
                -1.314 ,-1.32  ,-1.326 ,-1.333 ,-1.339 ,-1.346 ,-1.352 ,-1.359 ,-1.365 ,-1.371,
                -1.377 ,-1.383 ,-1.389 ,-1.396 ,-1.402 ,-1.408 ,-1.414 ,-1.42  ,-1.426 ,-1.431,
                -1.437 ,-1.443 ,-1.448 ,-1.453 ,-1.459 ,-1.465 ,-1.47  ,-1.476 ,-1.481 ,-1.486,
                -1.49  ,-1.495 ,-1.5   ,-1.505 ,-1.509 ,-1.514 ,-1.518 ,-1.522 ,-1.526 ,-1.531,
                -1.535 ,-1.538 ,-1.542 ,-1.546 ,-1.55  ,-1.553 ,-1.557 ,-1.56  ,-1.563 ,-1.567,
                -1.57  ,-1.573 ,-1.576 ,-1.579 ,-1.581 ,-1.584 ,-1.587 ,-1.589 ,-1.592 ,-1.594,
                -1.597 ,-1.599 ,-1.601 ,-1.603 ,-1.605 ,-1.607 ,-1.609 ,-1.611 ,-1.613 ,-1.614,
                -1.616 ,-1.618 ,-1.619 ,-1.621 ,-1.622 ,-1.624 ,-1.625 ,-1.626 ,-1.627 ,-1.629,
                -1.63  ,-1.631 ,-1.632 ,-1.633 ,-1.634 ,-1.635 ,-1.636 ,-1.636 ,-1.637 ,-1.638,
                -1.639 ,-1.639 ,-1.64  ,-1.64  ,-1.641 ,-1.642 ,-1.642 ,-1.643 ,-1.643 ,-1.644,
            ]

    for j in pbar:
        z = torch.randn(batch_size, model.in_channels, latent_size, latent_size, device=device)
        z = z.to(device)
        labels_gen = torch.tensor([local_rank * iterations + j] * batch_size)
        y = torch.Tensor(labels_gen).long().cuda()

        if args.search_each:
            oss_steps = search_OSS(wrap_model, z, z.shape[0], y, device, teacher_steps=args.teacher_steps, student_steps=args.student_steps, model_kwargs=model_kwargs)[0]
            print(f"OSS steps: {oss_steps}")

        samples_oss = infer_OSS(oss_steps, wrap_model, z, y, device, renorm_flag=args.renorm_flag, max_amp=max_amp, min_amp=min_amp, model_kwargs=model_kwargs)

        # teacher samples
        teacher_steps = list(range(1, args.teacher_steps+1))
        samples_tea = infer_OSS(teacher_steps, wrap_model, z, y, device, model_kwargs=model_kwargs)


        samples_oss = vae.decode(samples_oss / 0.18215).sample
        samples_tea = vae.decode(samples_tea / 0.18215).sample

        samples = torch.cat([samples_oss, samples_tea], dim=-1)
        samples = torch.clamp(127.5 * samples + 128.0, 0, 255).permute(0, 2, 3, 1).to("cpu", dtype=torch.uint8).numpy()



        for i, sample in enumerate(samples):
            
            if args.add_tag:
                h,w,c = sample.shape
                font = ImageFont.load_default()
                font = font.font_variant(size=30)
                pil_image = Image.fromarray(sample)
                draw = ImageDraw.Draw(pil_image)
                draw.text((10, 10), "OSS", font=font, fill='brown')
                draw.text((10 + w//2, 10), "Teacher-DiT", font=font, fill='brown')

                sample = np.array(pil_image)

            index = (local_rank * iterations + j) * batch_size + i
            Image.fromarray(sample).save(f"{sample_folder_dir}/{index:06d}.png")

        dist.barrier()






if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument("--model", type=str, choices=list(DiT_models.keys()), default="DiT-XL/2")
    parser.add_argument("--vae",  type=str, choices=["ema", "mse"], default="ema")
    parser.add_argument("--sample-dir", type=str, default="samples")
    parser.add_argument("--folder_name", type=str, default="folder_name")
    # parser.add_argument("--num-fid-samples", type=int, default=50_000)
    parser.add_argument("--image-size", type=int, choices=[256, 512], default=256)
    parser.add_argument("--num-classes", type=int, default=1000)
    parser.add_argument("--cfg-scale",  type=float, default=1.5)
    parser.add_argument("--global-seed", type=int, default=0)
    parser.add_argument("--tf32", action="store_true", default=True,
                        help="By default, use TF32 matmuls. This massively accelerates sampling on Ampere GPUs.")
    parser.add_argument("--ckpt", type=str, default=None,
                        help="Optional path to a DiT checkpoint (default: auto-download a pre-trained DiT-XL/2 model).")
    
    parser.add_argument('--search_batch', type=int, default=256)
    parser.add_argument('--teacher_steps', type=int, default=200)
    parser.add_argument('--student_steps', type=int, default=10)
    parser.add_argument("--batch_search", action="store_true", default=False,
                        help="whether use the batch search algorithm")
    parser.add_argument("--renorm_flag", action="store_true", default=False,
                        help="whether use the renorm during inference")
    parser.add_argument("--search_each", action="store_true", default=False,
                        help="whether search each image")
    parser.add_argument("--add_tag", action="store_true", default=False,
                        help="whether add tag in result image")


    args = parser.parse_args()
    main(args)
