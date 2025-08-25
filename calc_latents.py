from pipeline_runner import get_pipline_runner
from util.config import *
from util.utils import *
from util.image_filters import get_filters
from tqdm import trange
import os

conf = get_config()
print("Config:", OmegaConf.to_container(conf, resolve=True, throw_on_missing=False))

runner = get_pipline_runner(get_section(conf, "pipeline"))

output_path = conf.output_path

filters = get_filters(get_section(conf, "filters"))

for i in trange(conf.start, conf.end):
    img, meta = load_image(output_path, i)
    results = {
        "meta": meta,
        "og_latents": meta["latents"],
        "rev_latents": runner.invert_image(img)
    }

    for name, f in filters:
        # Refresh the pipeline each time, because for some reason it isn't repeatable.
        runner = get_pipline_runner(get_section(conf, "pipeline"))
        results[name] = runner.invert_image(f(img))

    torch.save(results, os.path.join(output_path, f"latents_{i:05}.pt"))
