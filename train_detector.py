import torch
from torchvision.transforms import functional as F

from src.logger import logger
from src.transform import TianchiOCRDynamicResize, TianchiOCRClip, TianchiPolygonsToBBoxes
from src.dataset import TianchiOCRDataset, TianchiOCRDataLoader
from src.fpn_densenet import FPNDenseNet
from src.rpn import RPNLoss
from src.config import RPN_ANCHOR_SCALES, RPN_ANCHOR_RATIOS, RPN_ANCHOR_MATCH_THRESH_HI, RPN_ANCHOR_MATCH_THRESH_LO, \
    RPN_NUM_POS_SAMPLES, RPN_NUM_NEG_SAMPLES, RPN_BBOX_DELTA_STD_DEV

N_MAX_EPOCHS = 1

# dataset = TianchiOCRDataset('/Users/rlan/datasets/ICPR/train_1000/image_1000', '/Users/rlan/datasets/ICPR/train_1000/txt_1000')
transforms = [TianchiOCRDynamicResize(divisible_by=32), TianchiOCRClip(), TianchiPolygonsToBBoxes()]
dataset = TianchiOCRDataset('/home/rlan/datasets/ICPR/train_1000/image_1000',
                            '/home/rlan/datasets/ICPR/train_1000/txt_1000',
                            transforms=transforms)
loader = TianchiOCRDataLoader(dataset, shuffle=False)

net = FPNDenseNet(num_ratios=len(RPN_ANCHOR_RATIOS))
if torch.cuda.is_available():
    net = net.cuda()

optimizer = torch.optim.Adam(net.parameters(), lr=1e-4)

for epoch in range(N_MAX_EPOCHS):
    for i, (im, label) in enumerate(loader):
        # logger.info('im.shape: {}'.format(im.size))
        # if im.size[0] > 700:
        #     continue
        im = F.to_tensor(im.convert('RGB')).view(1, 3, *im.size)
        gt_positions = torch.from_numpy(label[0])
        rpn_bbox_delta_std_dev = torch.Tensor(RPN_BBOX_DELTA_STD_DEV)
        rpn_cls_loss = torch.Tensor([0])
        rpn_bbox_loss = torch.Tensor([0])

        if torch.cuda.is_available():
            im = im.cuda()
            gt_positions = gt_positions.cuda()
            rpn_bbox_delta_std_dev = rpn_bbox_delta_std_dev.cuda()
            rpn_cls_loss = rpn_cls_loss.cuda()
            rpn_bbox_loss = rpn_bbox_loss.cuda()

        rpn_proposals = net(im)
        optimizer.zero_grad()

        fpn_fmap_shapes = [net.fpn1.shape, net.fpn2.shape, net.fpn3.shape, net.fpn4.shape]

        for p, fmap_shape, scale in zip(rpn_proposals, fpn_fmap_shapes, RPN_ANCHOR_SCALES):
            downsampled_rate = [f / r for f, r in zip(fmap_shape, im.shape)]
            assert downsampled_rate[2] == downsampled_rate[3]
            rpn_loss = RPNLoss(fmap_shape[2:], downsampled_rate[3], [scale], RPN_ANCHOR_RATIOS,
                               RPN_ANCHOR_MATCH_THRESH_HI, RPN_ANCHOR_MATCH_THRESH_LO, RPN_NUM_POS_SAMPLES,
                               RPN_NUM_NEG_SAMPLES, rpn_bbox_delta_std_dev)
            cl, bl = rpn_loss(p, gt_positions)
            rpn_cls_loss += cl
            rpn_bbox_loss += bl

        (rpn_cls_loss + rpn_bbox_loss).backward()
        logger.info('[epoch {}, iter {}] rpn_cls_loss: {:.4f}, '
                    'rpn_bbox_loss: {:.4f}'.format(epoch, i, rpn_cls_loss.cpu().detach().numpy()[0],
                                                   rpn_bbox_loss.cpu().detach().numpy()[0]))
        optimizer.step()
pass
