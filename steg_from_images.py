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
output_path = conf.output_path
os.makedirs(output_path, exist_ok=True)

print("Reading images")
wm_avg = np.zeros((runner.image_height, runner.image_width, 3), dtype=np.float64)
count = 0
for i in trange(conf.start, conf.end):
    no_wm_img, no_wm_meta = load_image(no_wm_path, i)
    wm_img, wm_meta = load_image(wm_path, i)

    compare_meta_and_state(no_wm_meta, wm_meta, wm.get_state())

    wm_avg += np.asarray(wm_img).astype(np.float64) - np.asarray(no_wm_img).astype(np.float64)
    count += 1

print("Averaging")
# average all of the watermarked images
wm_avg /= count
wm_avg_img = PIL.Image.fromarray(np.uint8(np.clip(wm_avg + 127, 0, 255)))
wm_avg_img.save(os.path.join(output_path, "wm_avg.png"))

print("Testing steg removal attack")
for i in trange(conf.start, conf.end):
    no_wm_img, no_wm_meta = load_image(no_wm_path, i)
    wm_img, wm_meta = load_image(wm_path, i)

    compare_meta_and_state(no_wm_meta, wm_meta, wm.get_state())

    wm_removed_image = PIL.Image.fromarray(np.uint8(np.clip(np.asarray(wm_img) - wm_avg, 0, 255)))
    wm_removed_image.save(os.path.join(output_path, f"wm_removed_{i:04}.png"))
    
    results = [i]
    results.append(score_image(runner, wm, no_wm_img))
    results.append(score_image(runner, wm, wm_removed_image))
    print(",".join((str(i) for i in results)))
