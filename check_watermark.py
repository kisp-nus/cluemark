import PIL
from stable_diffusion.pipeline_runner import *
from watermark import get_watermark_from_conf
from util.image_filters import *
from util.config import *
from tqdm import trange
import os

def load_image(path, i):
    path = os.path.join(path, f"{i:05}")
    meta = torch.load(path + ".pt")
    with PIL.Image.open(path + ".png") as img:
        return img.copy(), meta

def score_image(runner, wm, filter, img):
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

conf = get_config()
print("Config:", OmegaConf.to_container(conf, resolve=True, throw_on_missing=False))

runner = PipelineRunner(get_section(conf, "pipeline"))
wm = get_watermark_from_conf(get_section(conf, "watermark"), runner._pipe, conf.device)
if not wm:
    raise ValueError("Must have a watermark")

no_wm_path = conf.no_wm_path
wm_path = conf.output_path

filters = get_filters(get_section(conf, "filters"))

print(",".join(["i"] + [ f"{name}_no_wm,{name}_wm" for name, _ in filters ]))

for i in trange(conf.start, conf.end):
    no_wm_img, no_wm_meta = load_image(no_wm_path, i)
    wm_img, wm_meta = load_image(wm_path, i)

    if no_wm_meta['seed'] != wm_meta['seed']:
        print(f"WARNING! Seed doesn't match, {no_wm_meta['seed']} vs {wm_meta['seed']}")
    if no_wm_meta['prompt'] != wm_meta['prompt']:
        print(f"WARNING! Prompt doesn't match, {no_wm_meta['prompt']} vs {wm_meta['prompt']}")
    if wm_meta['watermark'].keys() != wm.get_state().keys():
        print("WARNING! WM meta and config keys don't match")
        print(wm_meta['watermark'].keys())
        print(wm.get_state().keys())
    else:
        for k in wm_meta['watermark'].keys():
            if compare_maybe_tensor(wm_meta['watermark'][k], wm.get_state()[k]):
                print("WARNING! WM meta and config don't match on", k)
                print("WM meta", wm_meta['watermark'][k])
                print("config", wm.get_state()[k])

    results = [i]
    for _, f in filters:
        results.append(score_image(runner, wm, f, no_wm_img))
        results.append(score_image(runner, wm, f, wm_img))

    print(",".join((str(i) for i in results)))
