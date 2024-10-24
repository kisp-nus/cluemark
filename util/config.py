# Helper to get config from command line and yaml files

from omegaconf import OmegaConf
import os, sys
import torch

def get_config(args=None):
    if args is None:
        args = sys.argv[1:]
    
    conf = OmegaConf.create()
    if len(args) < 1:
        return conf
    includes = [args[0]]
    while len(includes) > 0:
        filename = includes.pop(0)
        new_conf = OmegaConf.load(filename)
        includes += new_conf.get("include", [])
        new_conf.pop("include", None)
        conf = OmegaConf.merge(new_conf, conf)
    if len(args) > 1:
        conf = OmegaConf.merge(conf, OmegaConf.from_dotlist(args[1:]))
    return conf

def get_section(conf, section):
    return conf.get(section, OmegaConf.create())

def get_dtype(s):
    if s == "float32" or s == "float" or s =="single" or s == "f32":
        return torch.float32
    if s == "float64" or s == "double" or s == "f64":
        return torch.float64
    if s == "float16" or s == "half" or s == "f16":
        return torch.float16
    if s == "bfloat16" or s == "bf16":
        return torch.bfloat16

def save_config(conf, path):
    OmegaConf.save(OmegaConf.to_container(conf, resolve=True, throw_on_missing=False), path)
