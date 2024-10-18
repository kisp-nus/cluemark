import torch
from scipy import stats, fft

def get_dct(t):
    return torch.tensor(fft.dct(t.numpy(force=True)), device=t.device, dtype=t.dtype)

def inv_dct(t):
    return torch.tensor(fft.idct(t.numpy(force=True)), device=t.device, dtype=t.dtype)

def uniform_cdf(x):
    return stats.uniform.cdf(x, -0.5, 1.0)

def get_ks_score(samples, secret_direction, gamma, beta=0):
    ze_samples = samples + get_random_samples(samples.size(), var=beta) if beta > 0 else samples
    errs = get_hclwe_scores(ze_samples, secret_direction, gamma)
    return stats.kstest(errs, uniform_cdf)

def get_random_samples(dims, var=1/4, device="cpu", dtype=torch.float32):
    return torch.randn(*dims, device=device, dtype=dtype) * var

def sample_unit_vector(dim, device="cpu", dtype=torch.float32):
    result = get_random_samples((dim,), device=device, dtype=dtype)
    return result / torch.linalg.norm(result)

def get_dims(secret_dim, sample_dims):
    total_dim = torch.prod(torch.tensor(sample_dims))
    if total_dim % secret_dim != 0: raise ValueError("secret dim must divide total sample dims")
    n_samples = total_dim // secret_dim
    return total_dim, n_samples
    
def get_hclwe_samples(secret_direction, dims, gamma, beta, var=1/4, device="cpu", dtype=torch.float32):
    secret_dim = len(secret_direction)
    total_dim, n_samples = get_dims(secret_dim, dims)

    # use rejection sample to condition on z being within beta
    samples = []
    while len(samples) < n_samples:
        x = get_random_samples((secret_dim,), var=var, device=device, dtype=dtype)
        if abs(gamma * torch.dot(x, secret_direction)) % 1 < beta:
            samples.append(x)

    return torch.stack(samples).reshape(dims)

def get_hclwe_errors(samples, secret_direction, gamma):
    secret_dim = len(secret_direction)
    total_dim, n_samples = get_dims(secret_dim, samples.size())
    return (gamma * (samples.view((n_samples, secret_dim)) @ secret_direction) + 0.5) % 1 - 0.5

def hclwe_score_stdev(samples, secret_direction, gamma):
    return get_hclwe_errors(samples, secret_direction, gamma).std()

def hclwe_score(samples, secret_direction, gamma):
    return stats.kstest(get_hclwe_errors(samples, secret_direction, gamma).cpu().numpy(), uniform_cdf).statistic


def project_to_clwe(secret_direction, samples, gamma):
    secret_dim = len(secret_direction)
    total_dim, n_samples = get_dims(secret_dim, samples.size())
    inner_prod = samples.view((n_samples, secret_dim)) @ secret_direction
    k = torch.round(gamma * inner_prod)
    errors = k / gamma - inner_prod
    return (samples.view((n_samples, secret_dim)) + errors.view(-1, 1) @ secret_direction.view(1, -1)).reshape(samples.size())

def apply_wm_dct(secret_direction, samples, gamma):
    clwe_samples_dct = project_to_clwe(secret_direction, get_dct(samples), gamma)
    return inv_dct(clwe_samples_dct)

def get_wm_score(samples, secret_direction, gamma):
    errs = get_hclwe_errors(get_dct(samples), secret_direction, gamma)
    return stats.kstest(errs.numpy(force=True), uniform_cdf).statistic

def test_multi_dims(secret_dim, dims, gamma, beta):
    secret_direction = sample_unit_vector(secret_dim)
    raw_samples = get_random_samples(dims, var=1)
    print("raw sample score", hclwe_score(raw_samples, secret_direction, gamma))
    new_samples = project_to_clwe(secret_direction, raw_samples, gamma)
    print("new sample score", hclwe_score(new_samples, secret_direction, gamma))
    new_samples += get_random_samples(new_samples.size(), var=beta)
    print("after noise score", hclwe_score(new_samples, secret_direction, gamma))
    return raw_samples, new_samples

if __name__ == "__main__":
    import sys
    gamma = 0.5
    beta = 0.25
    secret_dim = 64
    if len(sys.argv) > 1:
        gamma = float(sys.argv[1])
    if len(sys.argv) > 2:
        beta = float(sys.argv[2])
    if len(sys.argv) > 3:
        secret_dim = int(sys.argv[3])

    test_multi_dims(64, (4, 64, 64), gamma, beta)

