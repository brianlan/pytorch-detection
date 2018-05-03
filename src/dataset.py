from pathlib import Path

import cv2
from PIL import Image
import torch
import numpy as np
from torch.utils.data import DataLoader

from .logger import logger


class ImageReadingError(Exception):
    pass


class ImageReader(object):
    def __call__(self, path):
        try:
            return Image.open(path)
        except Exception as e:
            raise ImageReadingError(e)


class TianchiOCRLabelReader(object):
    def __call__(self, path):
        with open(path, 'rb') as f:
            label = [l.strip().decode('utf-8').split(',') for l in f.readlines()]
            label = [l[:8] + (l[8:] if len(l) <= 9 else [''.join(l[8:])]) for l in label]
            label = np.array(label)
        return np.array(label[:, :-1], dtype=np.float_), np.array(label[:, -1])


class DetectionDataPathIter(object):
    def __init__(self, im_dir, im_suffix, label_dir, label_suffix, indices=None):
        self.im_dir, self.label_dir = Path(str(im_dir)), Path(str(label_dir))
        self.im_suffix, self.label_suffix = '.' + im_suffix.strip('.'), '.' + label_suffix.strip('.')
        self.indices = indices or [p.relative_to(self.label_dir).stem for p in self.label_dir.glob('**/*' + self.label_suffix)]

    def __len__(self):
        return len(self.indices)

    def __getitem__(self, idx):
        return str(self.im_dir / self.indices[idx]) + self.im_suffix, \
                str(self.label_dir / self.indices[idx]) + self.label_suffix


class DetectionDataset(torch.utils.data.Dataset):
    @property
    def data_path_iter(self):
        raise NotImplementedError
    
    @property
    def im_read(self):
        raise NotImplementedError
    
    @property
    def label_read(self):
        raise NotImplementedError

    def __getitem__(self, idx):
        im_path, label_path = self.data_path_iter[idx]
        return self.im_read(im_path), self.label_read(label_path)

    def __len__(self):
        return len(self.data_path_iter)


class TianchiOCRDataset(DetectionDataset):
    def __init__(self, im_dir, label_dir, indices=None):
        super().__init__()
        self._data_path_iter = DetectionDataPathIter(im_dir, '.jpg', label_dir, '.txt', indices=indices)
        self._im_reader = ImageReader()
        self._label_reader = TianchiOCRLabelReader()
    
    @property
    def data_path_iter(self):
        return self._data_path_iter
    
    @property
    def im_read(self):
        return self._im_reader
    
    @property
    def label_read(self):
        return self._label_reader


class TianchiOCRDataLoader(torch.utils.data.DataLoader):
    def __init__(self, dataset, shuffle=False, num_workers=0):
        super().__init__(dataset, batch_size=1, shuffle=shuffle, num_workers=num_workers)

    def __iter__(self):
        if self.num_workers == 0:
            for idx in self.batch_sampler:
                try:
                    im, label = self.dataset[idx[0]]  # batch_size is fixed to be 1, so we can directly use idx[0].
                except FileNotFoundError as e:
                    logger.info('File not found during data loading phase. err_msg: {}'.format(e))
                    continue
                except ImageReadingError as e:
                    logger.info('Error found when reading image. err_msg: {}'.format(e))
                    continue
                yield im, label
            return
