from clwe.clwe_watermark import get_clwe_watermark_from_conf
from tree_ring.tree_ring_watermark import TreeRingWatermark
from gaussian_shading.gaussian_shading import get_gaussian_shading_from_conf

def get_watermark_from_conf(conf, pipe, device):
    if not conf or conf.get("type", "none") == "none":
        return None
    if conf.type == "clwe":
        return get_clwe_watermark_from_conf(conf)
    if conf.type == "tree_ring":
        return TreeRingWatermark(conf, pipe, device)
    if conf.type == "gaussian_shading":
        return get_gaussian_shading_from_conf(conf, device)
    else:
        raise ValueError("Unsupported watermark type: " + conf.type)
