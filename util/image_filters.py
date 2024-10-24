import tempfile
from PIL import Image, ImageFilter
from torchvision import transforms

def jpeg_filter(img, ratio):
    with tempfile.TemporaryFile() as fp:
        img.save(fp, "jpeg", quality=ratio)
        with Image.open(fp, formats=["jpeg"]) as img2:
            return img2.copy()


ALL_FILTERS = {
    "none": lambda conf: ("clean", lambda img: img),
    "jpeg": lambda conf: (f"jpeg_{conf.quality}",
                          lambda img: jpeg_filter(img, conf.quality)),

    "rotate": lambda conf: (f"rotate_{conf.degrees}",
                            lambda img: transforms.RandomRotation((conf.degrees, conf.degrees))(img)),

    "crop": lambda conf: (f"crop_{conf.crop}_{conf.scale}",
                          lambda img: transforms.RandomResizedCrop(img.size, scale=(conf.scale, conf.scale),
                                                   ratio=(conf.crop, conf.crop))(img)),

    "brightness": lambda conf: (f"brightness_{conf.factor}",
                                lambda img: transforms.ColorJitter(brightness=conf.factor)(img)),

    "blur": lambda conf: (f"blur_{conf.radius}",
                          lambda img: img.filter(ImageFilter.GaussianBlur(radius=conf.radius))),
}

def get_filters(conf):
    filters = []
    for c in conf:
        filters.append(ALL_FILTERS[c.type](c))
    return filters

