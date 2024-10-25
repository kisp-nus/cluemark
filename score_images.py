from torchmetrics.multimodal.clip_score import CLIPScore
from util.image_filters import *
from util.config import *
from util.utils import *
from tqdm import trange

def calculate_clip_score(images, prompts, model = "openai/clip-vit-base-patch32"):
    transform = transforms.Compose([transforms.ToTensor()])
    image_tensors = [ transform(img) for img in images ]
    metric = CLIPScore(model)
    return metric(image_tensors, prompts).item()

conf = get_config()
print("Config:", OmegaConf.to_container(conf, resolve=True, throw_on_missing=False))

image_path = conf.output_path

print("Reading images")
images = []
prompts = []
for i in trange(conf.start, conf.end):
    img, meta = load_image(image_path, i)
    images.append(img)
    prompts.append(meta["prompt"])

assert len(images) == len(prompts)

result = calculate_clip_score(images, prompts)
print(result)
