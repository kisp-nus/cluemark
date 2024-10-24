from stable_diffusion.pipeline_runner import *
from watermark import get_watermark_from_conf
from util.config import *
from datasets import load_dataset
from tqdm import trange
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

conf = get_config()
print("Config:", OmegaConf.to_container(conf, resolve=True, throw_on_missing=False))

dataset, prompt_key = get_dataset(conf.get("dataset", "Gustavosta/Stable-Diffusion-Prompts"))
runner = PipelineRunner(get_section(conf, "pipeline"))
wm = get_watermark_from_conf(get_section(conf, "watermark"), runner._pipe, conf.device)
output_path = conf.output_path
os.makedirs(output_path, exist_ok=True)
save_config(conf, os.path.join(output_path, "config.yaml"))

for i in trange(conf.start, conf.end):
    seed = i + conf.seed
    prompt = dataset[i][prompt_key]
    latents = runner.get_random_latents(seed)
    if wm:
        latents = wm.inject_watermark(latents)

    img = runner.sample_image(prompt, latents)
    img.save(os.path.join(output_path, f"{i:05}.png"))
    torch.save({
        "seed": seed,
        "prompt": prompt,
        "latents": latents,
        "conf": OmegaConf.to_container(conf, resolve=True, throw_on_missing=False),
        "watermark": wm.get_state() if wm else None,
    }, os.path.join(output_path, f"{i:05}.pt"))
