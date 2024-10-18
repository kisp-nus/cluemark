import os
import argparse
import wandb
import copy
from tqdm import tqdm, trange
from statistics import mean, stdev
from sklearn import metrics
import PIL

import torch

from inverse_stable_diffusion import InversableStableDiffusionPipeline
from diffusers import DPMSolverMultistepScheduler
import open_clip
from optim_utils import *
from io_utils import *
from clwe_dwt import *


def main(args):
    # load diffusion model
    device = args.device if args.device is not None else 'cuda' if torch.cuda.is_available() else 'cpu'
    print("device", device)

    scheduler = DPMSolverMultistepScheduler.from_pretrained(args.model_id, subfolder='scheduler')
    pipe = InversableStableDiffusionPipeline.from_pretrained(
        args.model_id,
        scheduler=scheduler,
        torch_dtype=torch.float16,
        revision='fp16',
        )
    pipe = pipe.to(device)

    tester_prompt = '' # assume at the detection time, the original prompt is unknown
    text_embeddings = pipe.get_text_embedding(tester_prompt)

    # ground-truth patch
    gt_patch = get_watermarking_pattern(pipe, args, device)

    output_path = args.output_path
    os.makedirs(output_path, exist_ok=True)

    print("Reading images")
    wm_avg = np.zeros((args.image_length, args.image_length, 3), dtype=np.float64)
    count = 0
    for i in tqdm(range(args.start, args.end)):
        with PIL.Image.open(os.path.join(output_path, f"no_wm_{i:04}.png")) as orig_image_no_w:
            with PIL.Image.open(os.path.join(output_path, f"wm_{i:04}.png")) as orig_image_w:
                wm_avg += np.asarray(orig_image_w).astype(np.float64) - np.asarray(orig_image_no_w).astype(np.float64)
        count += 1

    print("Averaging")
    # average all of the watermarked images
    wm_avg /= count
    wm_avg_img = PIL.Image.fromarray(np.uint8(np.clip(wm_avg + 127, 0, 255)))
    wm_avg_img.save(os.path.join(output_path, "wm_avg.png"))

    print("Now with WM removed")

    secret = None
    gamma = args.w_gamma

    for i in tqdm(range(args.start, args.end)):
        meta = np.load(os.path.join(output_path, f"metadata_{i:04}.npy"), allow_pickle=True)
        meta_dict = meta.item()
            
        if secret is not None and not ((secret == meta_dict.get('secret')).all()):
            print("WARNING! SECRET DOESN'T MATCH AT", i)
        secret = meta_dict.get('secret')
    
        if gamma != meta_dict.get('gamma'):
            print("WARNING! GAMMA DOESN'T MATCH AT", i)

        # reverse img with watermarking but removed
        with PIL.Image.open(os.path.join(output_path, f"wm_{i:04}.png")) as orig_image_w:
            wm_removed_image = PIL.Image.fromarray(np.uint8(np.clip(np.asarray(orig_image_w) - wm_avg, 0, 255)))
        wm_removed_image.save(os.path.join(output_path, f"wm_removed_{i:04}.png"))
        img_w = transform_img(wm_removed_image).unsqueeze(0).to(text_embeddings.dtype).to(device)
        image_latents_w = pipe.get_image_latents(img_w, sample=False)

        reversed_latents_w = pipe.forward_diffusion(
            latents=image_latents_w,
            text_embeddings=text_embeddings,
            guidance_scale=1,
            num_inference_steps=args.test_num_inference_steps,
        )

        # eval
        w_metric = get_wm_dwt_score(reversed_latents_w, meta_dict.get('secret'), meta_dict.get('gamma'))
        print(i, meta_dict.get("no_wm_score"), meta_dict.get("wm_score"), w_metric)



if __name__ == '__main__':
    parser = argparse.ArgumentParser(description='diffusion watermark')
    parser.add_argument('--device', default=None, type=str)
    parser.add_argument('--output_path', default='./outputs/')
    parser.add_argument('--dataset', default='Gustavosta/Stable-Diffusion-Prompts')
    parser.add_argument('--start', default=0, type=int)
    parser.add_argument('--end', default=10, type=int)
    parser.add_argument('--image_length', default=512, type=int)
    parser.add_argument('--model_id', default='stabilityai/stable-diffusion-2-1-base')
    parser.add_argument('--num_images', default=1, type=int)
    parser.add_argument('--guidance_scale', default=7.5, type=float)
    parser.add_argument('--num_inference_steps', default=50, type=int)
    parser.add_argument('--test_num_inference_steps', default=None, type=int)
    parser.add_argument('--gen_seed', default=0, type=int)

    # watermark
    parser.add_argument('--w_seed', default=999999, type=int)
    parser.add_argument('--w_secret_dim', default=16, type=int)
    parser.add_argument('--w_gamma', default=0.4, type=float)
    parser.add_argument('--w_channel', default=0, type=int)
    parser.add_argument('--w_pattern', default='rand')
    parser.add_argument('--w_mask_shape', default='circle')
    parser.add_argument('--w_radius', default=10, type=int)
    parser.add_argument('--w_measurement', default='l1_complex')
    parser.add_argument('--w_injection', default='complex')
    parser.add_argument('--w_pattern_const', default=0, type=float)
    
    args = parser.parse_args()
    if args.test_num_inference_steps is None:
        args.test_num_inference_steps = args.num_inference_steps
    
    main(args)