def rpn_level_loss(pred_cls, pred_quads):
    fmap_size = pred_cls.shape[2:]


def rpn_loss(proposals, label):
    fpn1_cls, fpn1_reg, fpn2_cls, fpn2_reg, fpn3_cls, fpn3_reg, fpn4_cls, fpn4_reg = proposals
    return 1