import torch
import numpy as np
import pywt
from scipy import stats

def sample_unit_vector(secret_dim, seed=42):
    g = np.random.default_rng(seed)
    secret_dir = g.normal(0, 1, size=(secret_dim,))
    return secret_dir / np.linalg.norm(secret_dir)

def get_dims(secret_dim, sample_dims):
    total_dim = np.prod(sample_dims)
    if total_dim % secret_dim != 0: raise ValueError("secret dim must divide total sample dims")
    n_samples = total_dim // secret_dim
    return total_dim, n_samples

def project_to_clwe(secret_direction, samples, gamma):
    secret_dim = len(secret_direction)
    total_dim, n_samples = get_dims(secret_dim, samples.shape)
    inner_prod = samples.reshape((n_samples, secret_dim)) @ secret_direction
    k = np.round(gamma * inner_prod)
    errors = k / gamma - inner_prod
    return (samples.reshape((n_samples, secret_dim))
            + errors.reshape(-1, 1) @ secret_direction.reshape(1, -1)).reshape(samples.shape)

def get_hclwe_errors(samples, secret_direction, gamma):
    secret_dim = len(secret_direction)
    total_dim, n_samples = get_dims(secret_dim, samples.shape)
    return (gamma * (samples.reshape((n_samples, secret_dim)) @ secret_direction) + 0.5) % 1 - 0.5

def uniform_cdf(x):
    return stats.uniform.cdf(x, -0.5, 1.0)

def get_hclwe_score(samples, secret_direction, gamma):
    errs = get_hclwe_errors(samples, secret_direction, gamma)
    return stats.kstest(errs, uniform_cdf).statistic

def get_dwt(samples):
    return pywt.dwt2(samples, 'haar')

def inv_dwt(ca, others):
    return pywt.idwt2((ca, others), 'haar')

def apply_wm_dwt(samples, secret_direction, gamma):
    cLL, (cLH, cHL, cHH) = get_dwt(samples.numpy(force=True))
    clwe_cLL = project_to_clwe(secret_direction, cLL, gamma)
    clwe_cLH = project_to_clwe(secret_direction, cLH, gamma)
    clwe_cHL = project_to_clwe(secret_direction, cHL, gamma)
    return torch.tensor(inv_dwt(clwe_cLL, (clwe_cLH, clwe_cHL, cHH)), device=samples.device, dtype=samples.dtype)

def get_wm_dwt_score(samples, secret_direction, gamma):
    cLL, (cLH, cHL, cHH) = get_dwt(samples.numpy(force=True))
    errs = np.concatenate((
        get_hclwe_errors(cLL, secret_direction, gamma),
        get_hclwe_errors(cLH, secret_direction, gamma),
        get_hclwe_errors(cHL, secret_direction, gamma)))
    return stats.kstest(errs, uniform_cdf).statistic
