import attr
import torch.nn as nn
import numpy as np

from anchors import AnchorGenerator, calc_anchor_match, get_delta
from loss import smooth_l1_loss, softmax_loss


@attr.s
class RPNLoss(nn.Module):
    fmap_shape = attr.ib()
    fmap_downsampled_rate = attr.ib()
    scales = attr.ib()
    ratios = attr.ib()

    def forward(self, proposals, gt_boxes):
        anchors = AnchorGenerator.generate_anchors(self.fmap_shape, self.fmap_downsampled_rate, self.scales,
                                                   self.ratios)
        matches, assigned_gt_bbox = calc_anchor_match(anchors, gt_boxes)
        pos_sample_idx = matches[matches == 1]
        neg_sample_idx = matches[matches == 0]
        pred_cls, pred_bboxes = proposals
        cls_loss = softmax_loss(np.concatenate((pred_cls[pos_sample_idx], pred_cls[neg_sample_idx])),
                                [1] * len(pos_sample_idx) + [0] * len(neg_sample_idx))
        delta = get_delta(anchors[pos_sample_idx], assigned_gt_bbox[pos_sample_idx])
        bbox_loss = smooth_l1_loss(pred_bboxes[pos_sample_idx], delta)
        return cls_loss, bbox_loss
