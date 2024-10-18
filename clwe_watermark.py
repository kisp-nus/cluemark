import torch
import numpy as np
from scipy import fft, stats
import pywt


def sample_unit_vector(secret_dim, seed=42):
    g = np.random.default_rng(seed)
    secret_dir = g.normal(0, 1, size=secret_dim)
    return secret_dir / np.linalg.norm(secret_dir)

def vslice(start, step):
    return tuple( slice(x, x+y) for x, y in zip(start, step) )

def inc_index(index, block_dim, shape):
    for i in reversed(range(len(index))):
        index[i] += block_dim[i]
        if index[i] < shape[i]:
            return index
        index[i] = 0
    return None

def pad_ones(l, dim):
    return (1,) * (l - len(dim)) + dim

def split_blocks(ar, block_dim):
    if len(block_dim) > ar.ndim:
        raise ValueError("block has more dimensions than array")
    block_dim = pad_ones(ar.ndim, block_dim)
    for i in range(ar.ndim):
        if ar.shape[i] % block_dim[i] != 0:
            raise ValueError("Block dim does not divide array shape.")

    index = np.zeros(ar.ndim, dtype=int)
    while index is not None:
        yield ar[vslice(index, block_dim)]
        index = inc_index(index, block_dim, ar.shape)

def extract_blocks(ar, block_dim):
    return np.stack([ b.flatten() for b in split_blocks(ar, block_dim)])

def restack_blocks(blocks, block_dim, shape):
    if len(block_dim) > len(shape):
        raise ValueError("block has more dimensions than array")
    block_dim = pad_ones(len(shape), block_dim)
    for i in range(len(shape)):
        if shape[i] % block_dim[i] != 0:
            raise ValueError("Block dim does not divide array shape.")
    ar = np.empty(shape, dtype=blocks.dtype)
    index = np.zeros(ar.ndim, dtype=int)
    i = 0
    while index is not None:
        ar[vslice(index, block_dim)] = blocks[i].reshape(block_dim)
        index = inc_index(index, block_dim, shape)
        i += 1
    return ar

def inner_prod_with_secret(samples, secret_direction):
    return extract_blocks(samples, secret_direction.shape) @ secret_direction.flatten()

def project_to_clwe(samples, secret_direction, gamma):
    inner_prod = inner_prod_with_secret(samples, secret_direction)
    k = np.round(gamma * inner_prod)
    errors = k / gamma - inner_prod
    deltas = errors.reshape(-1, 1) @ secret_direction.reshape(1, -1)
    return samples + restack_blocks(deltas, secret_direction.shape, samples.shape)

def get_hclwe_errors(samples, secret_direction, gamma):
    inner_prod = inner_prod_with_secret(samples, secret_direction)
    return (gamma * inner_prod + 0.5) % 1 - 0.5

def uniform_cdf(x):
    return stats.uniform.cdf(x, -0.5, 1.0)

def get_hclwe_score(samples, secret_direction, gamma):
    return stats.kstest(get_hclwe_errors(samples, secret_direction, gamma),
                        uniform_cdf).statistic

def uniform_cdf(x):
    return stats.uniform.cdf(x, -0.5, 1.0)

def get_watermark_from_conf(conf):
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
        self.seed = conf.seed
        self.secret = sample_unit_vector(self.secret_dim, self.seed)
    
    def inject_watermark(self, latents_np):
        return project_to_clwe(latents_np, self.secret, self.gamma)

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
