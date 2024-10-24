from tree_ring.io_utils import *
from tree_ring.optim_utils import *


def eval_watermark(reversed_latents_w, watermarking_mask, gt_patch, args):
    if 'complex' in args.w_measurement:
        reversed_latents_w_fft = torch.fft.fftshift(torch.fft.fft2(reversed_latents_w), dim=(-1, -2))
        target_patch = gt_patch
    elif 'seed' in args.w_measurement:
        reversed_latents_w_fft = reversed_latents_w
        target_patch = gt_patch
    else:
        NotImplementedError(f'w_measurement: {args.w_measurement}')

    if 'l1' in args.w_measurement:
        w_metric = torch.abs(reversed_latents_w_fft[watermarking_mask] - target_patch[watermarking_mask]).mean().item()
    else:
        NotImplementedError(f'w_measurement: {args.w_measurement}')

    return w_metric


class TreeRingWatermark:
    def __init__(self, conf, pipe, device):
        self.device = device
        self.conf = conf
        self.gt_patch = get_watermarking_pattern(pipe, conf, device)

    def inject_watermark(self, latents):
        watermarking_mask = get_watermarking_mask(latents, self.conf, self.device)
        return inject_watermark(latents, watermarking_mask, self.gt_patch, self.conf)
    
    def check_watermark(self, latents):
        watermarking_mask = get_watermarking_mask(latents, self.conf, self.device)
        return eval_watermark(latents, watermarking_mask, self.gt_patch, self.conf)
    
    def get_state(self):
        return {
            "conf": self.conf,
            "gt_patch": self.gt_patch,
        }
