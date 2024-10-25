from torchmetrics.image.inception import InceptionScore
from torchmetrics.multimodal.clip_score import CLIPScore
from torchvision.transforms import PILToTensor
from util.image_filters import *
from util.config import *
from util.utils import *
from tqdm import trange

def calculate_clip_score(images, prompts, batch_size = 10, model = "openai/clip-vit-base-patch32"):
    assert len(images) == len(prompts)
    transform = transforms.Compose([transforms.ToTensor()])
    metric = CLIPScore(model)
    scores = []
    for i in range(0, len(images), batch_size):
        image_tensors = [ transform(img) for img in images[i : min(len(images), i + batch_size)] ]
        scores.append(metric(image_tensors, prompts[i : i + len(image_tensors)]).item())
    
    return torch.mean(torch.tensor(scores)).item()

def calculate_inception_score(images, batch_size = 10):
    transform = transforms.Compose([PILToTensor()])
    metric = InceptionScore()
    images_tensor = torch.stack([ transform(img) for img in images ])
    metric.update(images_tensor)
    return metric.compute()

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

print("clip", calculate_clip_score(images, prompts))
print("inception", calculate_inception_score(images))
