import torch
from oss_flux import SearchPipeline_Flux
import numpy as np
import torch.backends.cudnn as cudnn
import os
import argparse
import logging
import sys

#  You should login to use the flux model
# from huggingface_hub import login
# login(token="")

from PIL import ImageDraw, Image, ImageFont


def _init_logging():
    # logging

    logging.basicConfig(
        level=logging.INFO,
        format="[%(asctime)s] %(levelname)s: %(message)s",
        handlers=[logging.StreamHandler(stream=sys.stdout)])

def main(args):
    _init_logging()
    device = f"cuda"
    pipe = SearchPipeline_Flux.from_pretrained("black-forest-labs/FLUX.1-dev", torch_dtype=torch.bfloat16).to(device)

    os.makedirs(f'{args.save_imgs_dir}', exist_ok=True)
    
    print("teacher steps: ", args.teacher_steps)
    print("student steps: ", args.student_steps)
    out = pipe(
        prompt=args.prompt,
        guidance_scale=3.5,
        height=args.h,
        width=args.w,
        num_inference_steps=args.teacher_steps,
        student_steps = args.student_steps,
    ).images[0]

    sample = np.array(out)
    if args.add_tag:
        h,w,c = sample.shape
        font = ImageFont.load_default()
        font = font.font_variant(size=30)
        pil_image = Image.fromarray(sample)
        draw = ImageDraw.Draw(pil_image)
        draw.text((10, 10), "OSS", font=font, fill='brown')
        draw.text((10 + w//2, 10), "Teacher-FLUX", font=font, fill='brown')
        sample = np.array(pil_image)

    Image.fromarray(sample).save(f"{args.save_imgs_dir}/OSS_{args.student_steps}steps_result.png")

if __name__ == "__main__":


    cudnn.benchmark = True

    parser = argparse.ArgumentParser()

    parser.add_argument("--prompt", type=str)
    parser.add_argument("--save_intern_dir", type=str)
    parser.add_argument("--save_imgs_dir", type=str)
    
    parser.add_argument("--w", type=int)
    parser.add_argument("--h", type=int)
    parser.add_argument("--teacher_steps", type=int)
    parser.add_argument("--student_steps", type=int)
    parser.add_argument("--seed", type=int, default=0)
    parser.add_argument("--add_tag", action="store_true", default=False,
                        help="whether add tag in result image")

    args = parser.parse_args()
    torch.manual_seed(args.seed)
    np.random.seed(args.seed)
    main(args)
