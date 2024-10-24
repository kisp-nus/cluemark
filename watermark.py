from clwe_watermark import get_clwe_watermark_from_conf
from tree_ring.tree_ring_watermark import TreeRingWatermark

def get_watermark_from_conf(conf, pipe, device):
    if not conf or conf.get("type", "none") == "none":
        return None
    if conf.type == "clwe":
        return get_clwe_watermark_from_conf(conf)
    if conf.type == "tree_ring":
        return TreeRingWatermark(conf, pipe, device)
    else:
        raise ValueError("Unsupported watermark type: " + conf.type)
