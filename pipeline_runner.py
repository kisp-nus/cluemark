import torch
from torchvision import transforms
from inverse_stable_diffusion import InversableStableDiffusionPipeline
from diffusers import DPMSolverMultistepScheduler
import numpy as np
import random

def set_random_seed(seed=0):
    torch.manual_seed(seed + 0)
    torch.cuda.manual_seed(seed + 1)
    torch.cuda.manual_seed_all(seed + 2)
    np.random.seed(seed + 3)
    torch.cuda.manual_seed_all(seed + 4)
    random.seed(seed + 5)


def transform_img(image, target_size=512):
    tform = transforms.Compose(
        [
            transforms.Resize(target_size),
            transforms.CenterCrop(target_size),
            transforms.ToTensor(),
        ]
    )
    image = tform(image)
    return 2.0 * image - 1.0

class PipelineRunner:

    def __init__(self, conf) -> None:
        self.model_id = conf.model_id
        self.device = conf.device
        scheduler = DPMSolverMultistepScheduler.from_pretrained(self.model_id, subfolder='scheduler')
        self._pipe = InversableStableDiffusionPipeline.from_pretrained(
            self.model_id,
            scheduler=scheduler,
            torch_dtype=torch.float16,
            revision='fp16',
            )
        self._pipe = self._pipe.to(self.device)
        self.guidance_scale = conf.guidance_scale
        self.num_inference_steps = conf.num_inference_steps
        self.image_width = conf.image_width
        self.image_height = conf.get("image_height", self.image_width)
        # save empty text embeddings for inverting model,
        # since we assume that at retrieval we don't have the promp
        self.text_embeddings = self._pipe.get_text_embedding("")

    def get_random_latents(self, seed = None):
        if seed: set_random_seed(seed)
        return self._pipe.get_random_latents()
    
    def sample_image(self, prompt, latents):
        return self._pipe(
            prompt,
            num_images_per_prompt=1,
            guidance_scale=self.guidance_scale,
            num_inference_steps=self.num_inference_steps,
            height=self.image_width,
            width=self.image_height,
            latents=latents,
        ).images[0]
    
    def invert_image(self, img):
        trans_img = transform_img(img).unsqueeze(0).to(self.text_embeddings.dtype).to(self.device)
        image_latents = self._pipe.get_image_latents(trans_img, sample=False)
        return self._pipe.forward_diffusion(
            latents=image_latents,
            text_embeddings=self.text_embeddings,
            guidance_scale=1,
            num_inference_steps=self.num_inference_steps,
        )