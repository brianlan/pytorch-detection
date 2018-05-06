from pathlib import Path

import cv2
import numpy as np
import matplotlib.pyplot as plt
from tqdm import tqdm

from src.dataset import TianchiOCRDataset, TianchiOCRDataLoader


def norm(v):
    return np.sqrt((v * v).sum())


def calc_quadrilateral_w_h_ratios(q):
    w, h = np.abs(q[:, 4] - q[:, 0]), np.abs(q[:, 5] - q[:, 1])
    return w / (h + 1e-7)


# im_dir = Path('/Users/rlan/datasets/ICPR/train_1000/image_1000')
# label_dir = Path('/Users/rlan/datasets/ICPR/train_1000/txt_1000')
im_dir = Path('/home/rlan/datasets/ICPR/train_1000/image_1000')
label_dir = Path('/home/rlan/datasets/ICPR/train_1000/txt_1000')
dataset = TianchiOCRDataset(str(im_dir), str(label_dir))
loader = TianchiOCRDataLoader(dataset, shuffle=False)
save_dir = Path('/tmp/icpr')
im_shapes = []
w_h_ratios = []
for label_path, (im, label) in tqdm(zip(label_dir.glob('**/*.txt'), loader)):
    w_h_ratios.extend(calc_quadrilateral_w_h_ratios(label[0]).tolist())
    rel_path = label_path.relative_to(label_dir)
    save_path = (save_dir / rel_path).with_suffix('.png')
    save_path.parent.mkdir(parents=True, exist_ok=True)

    im_ = np.array(im)
    if len(im_.shape) < 3 or im_.shape[2] != 3:
        im_ = np.array(im.convert('RGB'))

    # for pts, txt in zip(*label):
    #     pts = pts.asnumpy().reshape(-1, 1, 2).astype(np.int32)
    #     if txt == '###':
    #         cv2.polylines(im, [pts], True, (0, 0, 0), 1)
    #         cv2.line(im, (pts[0][0][0], pts[0][0][1]), (pts[2][0][0], pts[2][0][1]), (0, 0, 0))
    #         cv2.line(im, (pts[1][0][0], pts[1][0][1]), (pts[3][0][0], pts[3][0][1]), (0, 0, 0))
    #     else:
    #         cv2.polylines(im, [pts], True, (255, 0, 0), 2)
    #
    # cv2.imwrite(str(save_path), cv2.cvtColor(im, cv2.COLOR_RGB2BGR))
    im_shapes.append(im_.shape)

im_shapes = np.array(im_shapes)
(im_shapes[:, 0] == im_shapes[:, 1]).sum() / len(im_shapes)
plt.hist(im_shapes[:, 0], bins=60)
plt.xlim(0, 1500)
plt.show()
pass