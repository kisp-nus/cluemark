from util.config import *
from metrics.fid_score import calculate_fid_given_paths
import os

conf = get_config()
print("Config:", OmegaConf.to_container(conf, resolve=True, throw_on_missing=False))

def get_num_workers(conf):
    if conf.get("num_workers") is not None:
        return conf.num_workers
    try:
        num_cpus = len(os.sched_getaffinity(0))
    except AttributeError:
        # os.sched_getaffinity is not available under Windows, use
        # os.cpu_count instead (which may not return the *available* number
        # of CPUs).
        num_cpus = os.cpu_count()

    return min(num_cpus, 8) if num_cpus is not None else 0

num_workers = get_num_workers(conf)
print(f"Using {num_workers} workers")

results_filename = conf.results_file
if results_filename.endswith(".txt"):
    results_filename = results_filename[:-4] + "-fid.txt"
else:
    results_filename += "-fid.txt"
print("Writing results to", results_filename)

fid_value = calculate_fid_given_paths(
    [conf.no_wm_path, conf.output_path],
    conf.get("batch_size", 50),
    conf.device,
    conf.get("fid_dims", 2048),
    num_workers
)

print("FID: ", fid_value, flush=True)

os.makedirs(os.path.dirname(results_filename), exist_ok=True)
with open(results_filename, "a") as results_file:
    print("# Config:", OmegaConf.to_container(conf, resolve=True, throw_on_missing=False), file=results_file)
    print("FID: ", fid_value, file=results_file)
