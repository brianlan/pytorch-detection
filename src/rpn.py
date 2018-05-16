import attr
import torch
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
    match_thresh_hi = attr.ib()
    match_thresh_lo = attr.ib()
    num_pos_samples = attr.ib()
    num_neg_samples = attr.ib()

    def __attrs_post_init__(self):
        super().__init__()

    def forward(self, proposals, gt_boxes):
        anchors = AnchorGenerator.generate_anchors(self.fmap_shape, self.fmap_downsampled_rate, self.scales,
                                                   self.ratios)
        anchors = torch.Tensor(anchors).cuda() if gt_boxes.is_cuda else torch.Tensor(anchors)
        pos_idx, neg_idx, assigned_gt_bbox = calc_anchor_match(anchors, gt_boxes, self.fmap_downsampled_rate,
                                                               match_thresh_hi=self.match_thresh_hi,
                                                               match_thresh_lo=self.match_thresh_lo)
        if len(pos_idx) > self.num_pos_samples:
            pos_idx = pos_idx[torch.randperm(len(pos_idx))[:self.num_pos_samples]]

        if len(neg_idx) > self.num_neg_samples:
            neg_idx = neg_idx[torch.randperm(len(neg_idx))[:self.num_neg_samples]]

        pred_cls, pred_bboxes = proposals
        pred_cls = pred_cls\
            .permute(2, 3, 1, 0)\
            .contiguous()\
            .view(pred_cls.shape[2] * pred_cls.shape[3] * pred_cls.shape[1] // 2, 2)
        pred_bboxes = pred_bboxes\
            .permute(2, 3, 1, 0)\
            .contiguous()\
            .view(pred_bboxes.shape[2] * pred_bboxes.shape[3] * pred_bboxes.shape[1] // 4, 4)

        cls_label = torch.cat((torch.ones(len(pos_idx), dtype=torch.long), torch.zeros(len(neg_idx), dtype=torch.long)))

        if torch.cuda.is_available():
            cls_label = cls_label.cuda()
        cls_loss = torch.nn.CrossEntropyLoss()(torch.cat((pred_cls[pos_idx], pred_cls[neg_idx])), cls_label)
        delta = get_delta(anchors[pos_idx], assigned_gt_bbox[pos_idx])
        bbox_loss = smooth_l1_loss(pred_bboxes[pos_idx], delta)
        return cls_loss, bbox_loss
