import torch
import numpy as np

from src.anchors import calc_overlap


def test_calc_overlap():
    query_boxes = torch.Tensor([[1, 1, 5, 5],
                               [10, 4, 14, 8],
                               [7, 8, 11, 12]])
    ref_boxes = torch.Tensor([[12, 1, 16, 5],
                              [8, 7, 12, 11]])
    gt_overlaps = np.array([[0, 0],
                            [6 / 44, 6 / 44],
                            [0, 16 / 34]])
    overlaps = calc_overlap(query_boxes, ref_boxes).numpy()
    np.testing.assert_almost_equal(overlaps, gt_overlaps)
