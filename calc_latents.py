from pipeline_runner import get_pipline_runner
from util.config import *
from util.utils import *
from tqdm import trange
import os

conf = get_config()
print("Config:", OmegaConf.to_container(conf, resolve=True, throw_on_missing=False))

runner = get_pipline_runner(get_section(conf, "pipeline"))

output_path = conf.output_path

for i in trange(conf.start, conf.end):
    img, meta = load_image(output_path, i)
    latents = runner.invert_image(img)

    torch.save({
        "orig_latents": meta["latents"],
        "rev_latents": latents
    }, os.path.join(output_path, f"latents_{i:05}.pt"))
