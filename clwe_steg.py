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

    # dataset
    dataset, prompt_key = get_dataset(args)

    tester_prompt = '' # assume at the detection time, the original prompt is unknown
    text_embeddings = pipe.get_text_embedding(tester_prompt)

    # ground-truth patch
    gt_patch = get_watermarking_pattern(pipe, args, device)

    output_path = args.output_path
    os.makedirs(output_path, exist_ok=True)

    no_wm_images = []
    wm_images = []
    reversed_latents_no_w_list = []

    # TODO: use seed
    secret = sample_unit_vector(args.w_secret_dim)
    gamma = args.w_gamma

    for i in tqdm(range(args.start, args.end)):
        seed = i + args.gen_seed
        
        current_prompt = dataset[i][prompt_key]
        
        ### generation
        # generation without watermarking
        set_random_seed(seed)
        init_latents_no_w = pipe.get_random_latents()
        outputs_no_w = pipe(
            current_prompt,
            num_images_per_prompt=args.num_images,
            guidance_scale=args.guidance_scale,
            num_inference_steps=args.num_inference_steps,
            height=args.image_length,
            width=args.image_length,
            latents=init_latents_no_w,
            )
        orig_image_no_w = outputs_no_w.images[0]
        no_wm_images.append(orig_image_no_w)
        orig_image_no_w.save(os.path.join(output_path, f"no_wm_{i:04}.png"))
        
        # generation with watermarking
        if init_latents_no_w is None:
            set_random_seed(seed)
            init_latents_w = pipe.get_random_latents()
        else:
            init_latents_w = copy.deepcopy(init_latents_no_w)

        print("Sampling hCLWE latents")
        init_latents_w = apply_wm_dwt(init_latents_no_w, secret, gamma)
        print("Initial no-watermark score:", get_wm_dwt_score(init_latents_no_w, secret, gamma))
        print("Initial hCLWE score:", get_wm_dwt_score(init_latents_w, secret, gamma))

        outputs_w = pipe(
            current_prompt,
            num_images_per_prompt=args.num_images,
            guidance_scale=args.guidance_scale,
            num_inference_steps=args.num_inference_steps,
            height=args.image_length,
            width=args.image_length,
            latents=init_latents_w,
            )
        orig_image_w = outputs_w.images[0]
        wm_images.append(orig_image_w)
        orig_image_w.save(os.path.join(output_path, f"wm_{i:04}.png"))

        # reverse img without watermarking
        img_no_w = transform_img(outputs_no_w.images[0]).unsqueeze(0).to(text_embeddings.dtype).to(device)
        image_latents_no_w = pipe.get_image_latents(img_no_w, sample=False)

        reversed_latents_no_w = pipe.forward_diffusion(
            latents=image_latents_no_w,
            text_embeddings=text_embeddings,
            guidance_scale=1,
            num_inference_steps=args.test_num_inference_steps,
        )
        reversed_latents_no_w_list.append(reversed_latents_no_w)

        # reverse img with watermarking
        img_w = transform_img(outputs_w.images[0]).unsqueeze(0).to(text_embeddings.dtype).to(device)
        image_latents_w = pipe.get_image_latents(img_w, sample=False)

        reversed_latents_w = pipe.forward_diffusion(
            latents=image_latents_w,
            text_embeddings=text_embeddings,
            guidance_scale=1,
            num_inference_steps=args.test_num_inference_steps,
        )

        # eval
        no_w_metric = get_wm_dwt_score(reversed_latents_no_w, secret, gamma)
        w_metric = get_wm_dwt_score(reversed_latents_w, secret, gamma)
        print(i, no_w_metric, w_metric)


    print("Averaging")
    # average all of the watermarked images
    wm_avg = np.zeros((args.image_length, args.image_length, 3))
    for no_wm_img, wm_img in tqdm(zip(no_wm_images, wm_images)):
        wm_avg += np.asarray(wm_img) - np.asarray(no_wm_img)
    wm_avg /= len(wm_images)
    print("wm min", np.min(wm_avg), "max", np.max(wm_avg), "avg", np.average(wm_avg))
    wm_avg_img = PIL.Image.fromarray(np.uint8(wm_avg))
    wm_avg_img.save(os.path.join(output_path, "wm_avg.png"))
    wm_avg -= np.average(wm_avg)

    print("Now with WM removed")

    for i in trange(len(wm_images)):
        # reverse img with watermarking but removed
        wm_removed_image = PIL.Image.fromarray(np.uint8(np.clip(np.asarray(wm_images[i]) - wm_avg, 0, 255)))
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
        no_w_metric = get_wm_dwt_score(reversed_latents_no_w_list[i], secret, gamma)
        w_metric = get_wm_dwt_score(reversed_latents_w, secret, gamma)
        print(i, no_w_metric, w_metric)



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