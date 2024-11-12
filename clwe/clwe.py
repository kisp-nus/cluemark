import numpy as np
from scipy import stats


def get_random_samples(dims, var=1/4):
    g = np.random.default_rng()
    return g.normal(0, var, size=dims)

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

def project_to_clwe(samples, secret_direction, gamma, beta=0):
    gammap = np.sqrt(beta*beta + gamma*gamma)
    inner_prod = inner_prod_with_secret(samples, secret_direction)
    k = np.round(gammap * inner_prod)
    errors = k * gamma / gammap
    if beta > 0:
        errors += get_random_samples(errors.shape, var=beta)
    errors = (errors / gammap) - inner_prod
    deltas = errors.reshape(-1, 1) @ secret_direction.reshape(1, -1)
    return samples + restack_blocks(deltas, secret_direction.shape, samples.shape)

def get_hclwe_errors(samples, secret_direction, gamma):
    inner_prod = inner_prod_with_secret(samples, secret_direction)
    return np.abs((gamma * inner_prod + 0.5) % 1 - 0.5)

def uniform_cdf(x):
    return stats.uniform.cdf(x, 0, 0.5)

def get_hclwe_score(samples, secret_direction, gamma):
    return stats.kstest(get_hclwe_errors(samples, secret_direction, gamma),
                        uniform_cdf).statistic
