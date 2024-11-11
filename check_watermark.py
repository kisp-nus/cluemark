import PIL
from pipeline_runner import get_pipline_runner
from watermark import get_watermark_from_conf
from util.image_filters import *
from util.config import *
from util.utils import *
from tqdm import trange
import os

conf = get_config()
print("Config:", OmegaConf.to_container(conf, resolve=True, throw_on_missing=False))

runner = get_pipline_runner(get_section(conf, "pipeline"))
wm = get_watermark_from_conf(get_section(conf, "watermark"), runner._pipe, conf.device)
if not wm:
    raise ValueError("Must have a watermark")

no_wm_path = conf.no_wm_path
wm_path = conf.output_path

filters = get_filters(get_section(conf, "filters"))

print("Writing results to", conf.results_file)
os.makedirs(os.path.dirname(conf.results_file), exist_ok=True)
with open(conf.results_file, "a") as results_file:
    print("# Config:", OmegaConf.to_container(conf, resolve=True, throw_on_missing=False), file=results_file)
    print(",".join(["i"] + [ f"{name}_no_wm,{name}_wm" for name, _ in filters ]), file=results_file)

    for i in trange(conf.start, conf.end):
        no_wm_img, no_wm_meta = load_image(no_wm_path, i)
        wm_img, wm_meta = load_image(wm_path, i)

        compare_meta_and_state(no_wm_meta, wm_meta, wm.get_state())
        
        results = [i]
        for _, f in filters:
            results.append(score_image(runner, wm, no_wm_img, f))
            results.append(score_image(runner, wm, wm_img, f))

        print(",".join((str(i) for i in results)), file=results_file, flush=True)
