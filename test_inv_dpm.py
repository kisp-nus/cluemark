import PIL
from stable_diffusion.pipeline_runner import *
from clwe_watermark import *
from util.image_filters import *
from util.config import *
from tqdm import trange
from datasets import load_dataset
import os

def get_dataset(dataset_name):
    if 'laion' in dataset_name:
        dataset = load_dataset(dataset_name)['train']
        prompt_key = 'TEXT'
    elif 'coco' in dataset_name:
        with open('fid_outputs/coco/meta_data.json') as f:
            dataset = json.load(f)
            dataset = dataset['annotations']
            prompt_key = 'caption'
    else:
        dataset = load_dataset(dataset_name)['test']
        prompt_key = 'Prompt'

    return dataset, prompt_key

def calc_mse(l1, l2):
    err = l2 - l1
    return torch.mean(err * err).item()

conf = get_config()
print("Config:", OmegaConf.to_container(conf, resolve=True, throw_on_missing=False))

dataset, prompt_key = get_dataset(conf.get("dataset", "Gustavosta/Stable-Diffusion-Prompts"))
runner = PipelineRunner(get_section(conf, "pipeline"))
wm = get_watermark_from_conf(get_section(conf, "watermark"))


for i in trange(conf.start, conf.end):
    seed = i + conf.seed
    current_prompt = dataset[i][prompt_key]
    reverse_prompt = current_prompt if conf.get("reverse_with_prompt", False) else ""

    ### Generation
    no_wm_latents = runner.get_random_latents(seed)
    no_wm_image = runner.sample_image(current_prompt, no_wm_latents)
    no_wm_reversed_latents = runner.invert_image(no_wm_image, reverse_prompt)
    results = [i, calc_mse(no_wm_latents, no_wm_reversed_latents),
              wm.check_watermark(no_wm_latents), wm.check_watermark(no_wm_reversed_latents)]

    if wm:
        wm_latents = wm.inject_watermark(no_wm_latents)
        wm_image = runner.sample_image(current_prompt, wm_latents)
        wm_reversed_latents = runner.invert_image(wm_image, reverse_prompt)
        results += [calc_mse(wm_latents, wm_reversed_latents),
                wm.check_watermark(wm_latents), wm.check_watermark(wm_reversed_latents)]

    print()
    print(",".join([str(x) for x in results]))
    print()
    print()
