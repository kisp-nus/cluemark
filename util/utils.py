import PIL
import torch
import numpy as np
import os

def load_image(path, i):
    path = os.path.join(path, f"{i:05}")
    meta = torch.load(path + ".pt")
    with PIL.Image.open(path + ".png") as img:
        return img.copy(), meta

def score_image(runner, wm, img, filter=None):
    if filter:
        img = filter(img)
    latents = runner.invert_image(img)
    return wm.check_watermark(latents)

def compare_maybe_tensor(a, b):
    if torch.is_tensor(a):
        return (a.cpu() != b.cpu()).any()
    if isinstance(a, np.ndarray):
        return (a != b).any()
    else:
        return a != b

def compare_meta_and_state(no_wm_meta, wm_meta, wm_state):
    if no_wm_meta['seed'] != wm_meta['seed']:
        print(f"WARNING! Seed doesn't match, {no_wm_meta['seed']} vs {wm_meta['seed']}")
    if no_wm_meta['prompt'] != wm_meta['prompt']:
        print(f"WARNING! Prompt doesn't match, {no_wm_meta['prompt']} vs {wm_meta['prompt']}")

    if wm_meta['watermark'].keys() != wm_state.keys():
        print("WARNING! WM meta and config keys don't match")
        print(wm_meta['watermark'].keys())
        print(wm_state.keys())
    else:
        for k in wm_meta['watermark'].keys():
            if compare_maybe_tensor(wm_meta['watermark'][k], wm_state[k]):
                print("WARNING! WM meta and config don't match on", k)
                print("WM meta", wm_meta['watermark'][k])
                print("config", wm_state[k])
