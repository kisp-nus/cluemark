import PIL
from stable_diffusion.pipeline_runner import *
from watermark import get_watermark_from_conf
from util.image_filters import *
from util.config import *
from util.utils import *
from tqdm import trange
import os

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

    compare_meta_and_state(no_wm_meta, wm_meta, wm.get_state())
    
    results = [i]
    for _, f in filters:
        results.append(score_image(runner, wm, no_wm_img, f))
        results.append(score_image(runner, wm, wm_img, f))

    print(",".join((str(i) for i in results)))
