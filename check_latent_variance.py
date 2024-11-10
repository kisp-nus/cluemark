import torch
from tqdm import trange
import os, sys

if len(sys.argv) != 5:
    print("Usage:", sys.argv[0], "[path] [start] [end] [device]")
    sys.exit(1)

folder = sys.argv[1]
start = int(sys.argv[2])
end = int(sys.argv[3])
device = sys.argv[4]

def get_mean_std(t):
    s, u = torch.std_mean(t)
    return u.item(), s.item()

print("i,orig_mean,orig_std,rev_mean,rev_std,diff_mean,diff_std")
for i in trange(start, end):
    d = torch.load(os.path.join(folder, f"latents_{i:05}.pt"), map_location=torch.device(device))
    results = [i]
    results += get_mean_std(d["orig_latents"])
    results += get_mean_std(d["rev_latents"])
    results += get_mean_std(d["rev_latents"] - d["orig_latents"])
    print(",".join([ str(x) for x in results]))
