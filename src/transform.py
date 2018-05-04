from math import ceil

import torchvision.transforms.functional as F


class TianchiOCRDynamicResize(object):
    def __init__(self, divisible_by=16):
        self.divider = divisible_by

    def __call__(self, img, label):
        size = img.size
        short_side = min(size)
        try_times = int(ceil(short_side / self.divider))
        for i in range(try_times + 2):
            if i * self.divider > short_side:
                new_size = (i - 1) * self.divider
                img = F.resize(img, (new_size, new_size))
                label[0][:, ::2] *= new_size / size[0]
                label[0][:, 1::2] *= new_size / size[1]
                return img, label