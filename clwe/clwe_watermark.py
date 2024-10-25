import torch
import numpy as np
from scipy import fft, stats
import pywt
from clwe.clwe import *

def get_clwe_watermark_from_conf(conf):
    if not conf or conf.get("type", "none") == "none":
        return None
    if conf.type != "clwe":
        raise ValueError("Unsupported watermark type: " + conf.type)
    
    impl = BaseCLWE(conf)
    if conf.get("dct", False):
        impl = DCTWrapper(impl)
    if len(conf.get("dwt_bands", [])) > 0:
        impl = DWTWrapper(conf, impl)
    return NumpyWatermark(impl)

class NumpyWatermark:
    def __init__(self, impl):
        self._impl = impl

    def inject_watermark(self, latents):
        return torch.tensor(self._impl.inject_watermark(latents.numpy(force=True)),
                            device=latents.device, dtype=latents.dtype)
    
    def check_watermark(self, latents):
        return stats.kstest(self._impl.get_errors(latents.numpy(force=True)),
                            uniform_cdf).statistic
    
    def get_state(self):
        return self._impl.get_state()


class BaseCLWE:
    def __init__(self, conf) -> None:
        self.secret_dim = conf.secret_dim
        self.gamma = conf.gamma
        self.beta = conf.beta
        self.seed = conf.seed
        self.secret = sample_unit_vector(self.secret_dim, self.seed)
    
    def inject_watermark(self, latents_np):
        return project_to_clwe(latents_np, self.secret, self.gamma, self.beta)

    def get_errors(self, latents_np):
        return get_hclwe_errors(latents_np, self.secret, self.gamma)
    
    def get_state(self):
        return {
            "secret_dim": self.secret_dim,
            "gamma": self.gamma,
            "seed": self.seed,
            "secret": self.secret,
        }

class DCTWrapper:
    def __init__(self, sub) -> None:
        self._sub = sub

    def inject_watermark(self, latents_np):
        return fft.idct(self._sub.inject_watermark(fft.dct(latents_np)))
    
    def get_errors(self, latents_np):
        return self._sub.get_errors(fft.dct(latents_np))
    
    def get_state(self):
        result = self._sub.get_state()
        result["dct"] = True
        return result

class DWTWrapper:
    def __init__(self, conf, sub) -> None:
        self.bands = conf.dwt_bands
        self._sub = sub

    def inject_watermark(self, latents_np):
        cLL, (cLH, cHL, cHH) = pywt.dwt2(latents_np, 'haar')
        if "LL" in self.bands:
            cLL = self._sub.inject_watermark(cLL)
        if "LH" in self.bands:
            cLH = self._sub.inject_watermark(cLH)
        if "HL" in self.bands:
            cHL = self._sub.inject_watermark(cHL)
        if "HH" in self.bands:
            cHH = self._sub.inject_watermark(cHH)
        
        return pywt.idwt2((cLL, (cLH, cHL, cHH)), 'haar')
    
    def get_errors(self, latents_np):
        cLL, (cLH, cHL, cHH) = pywt.dwt2(latents_np, 'haar')
        errors = []
        if "LL" in self.bands:
            errors.append(self._sub.get_errors(cLL))
        if "LH" in self.bands:
            errors.append(self._sub.get_errors(cLH))
        if "HL" in self.bands:
            errors.append(self._sub.get_errors(cHL))
        if "HH" in self.bands:
            errors.append(self._sub.get_errors(cHH))
        return np.concatenate(errors)

    def get_state(self):
        result = self._sub.get_state()
        result["dwt_bands"] = self.bands
        return result
