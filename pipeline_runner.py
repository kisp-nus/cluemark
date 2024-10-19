import torch
from torchvision import transforms
from inverse_stable_diffusion import InversableStableDiffusionPipeline
from diffusers import DPMSolverMultistepScheduler
from config import get_dtype
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
        scheduler = DPMSolverMultistepScheduler(
            beta_end=0.012,
            beta_schedule='scaled_linear',
            beta_start=0.00085,
            num_train_timesteps=1000,
            prediction_type="epsilon",
            steps_offset=1, 
            trained_betas=None,
            solver_order=conf.solver_order,
        )
        self._pipe = InversableStableDiffusionPipeline.from_pretrained(
            self.model_id,
            scheduler=scheduler,
            torch_dtype=get_dtype(conf.dtype),
            )
        self._pipe = self._pipe.to(self.device)
        self.guidance_scale = conf.guidance_scale
        self.num_inference_steps = conf.num_inference_steps
        self.image_width = conf.image_width
        self.image_height = conf.get("image_height", self.image_width)
        self.inv_order = conf.get("inv_order", conf.solver_order)

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
        )[0].images[0]

    def get_text_embeddings(self, prompt):
        text_embeddings_tuple = self._pipe.encode_prompt(
            prompt, self.device, 1, self.guidance_scale > 1.0, None
        )
        return torch.cat([text_embeddings_tuple[1], text_embeddings_tuple[0]])
    
    def invert_image(self, img, prompt=""):
        text_embeddings = self.get_text_embeddings(prompt)
        trans_img = transform_img(img).unsqueeze(0).to(text_embeddings.dtype).to(self.device)
        image_latents = self._pipe.decoder_inv(trans_img)
        return self._pipe.forward_diffusion(
            latents=image_latents,
            text_embeddings=text_embeddings,
            guidance_scale=self.guidance_scale,
            num_inference_steps=self.num_inference_steps,
            inverse_opt=(self.inv_order != 0),
            inv_order=self.inv_order
        )