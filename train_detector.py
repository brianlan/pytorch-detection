import torch
from torchvision.transforms import functional as F

from src.transform import TianchiOCRDynamicResize, TianchiOCRClip
from src.dataset import TianchiOCRDataset, TianchiOCRDataLoader
from src.fpn_densenet import FPNDenseNet
from src.rpn import RPNLoss
from src.config import RPN_ANCHOR_SCALES, RPN_ANCHOR_RATIOS

N_MAX_EPOCHS = 10

# dataset = TianchiOCRDataset('/Users/rlan/datasets/ICPR/train_1000/image_1000', '/Users/rlan/datasets/ICPR/train_1000/txt_1000')
transforms = [TianchiOCRDynamicResize(divisible_by=32), TianchiOCRClip()]
dataset = TianchiOCRDataset('/home/rlan/datasets/ICPR/train_1000/image_1000',
                            '/home/rlan/datasets/ICPR/train_1000/txt_1000',
                            transforms=transforms)
loader = TianchiOCRDataLoader(dataset, shuffle=False)

net = FPNDenseNet()
if torch.cuda.is_available():
    net = net.cuda()

optimizer = torch.optim.Adam(net.parameters(), lr=1e-4)

for epoch in range(N_MAX_EPOCHS):
    for im, label in loader:
        im = F.to_tensor(im).view(1, 3, *im.size)
        gt_positions = torch.from_numpy(label[0])

        if torch.cuda.is_available():
            im = im.cuda()
            gt_positions = gt_positions.cuda()

        rpn_proposals = net(im)
        optimizer.zero_grad()

        fpn_fmap_shapes = [net.fpn1.shape, net.fpn2.shape, net.fpn3.shape, net.fpn4.shape]
        for p, shape, scale in zip(rpn_proposals, fpn_fmap_shapes, RPN_ANCHOR_SCALES):
            downsampled_rate = shape / im.shape
            cls_loss, bbox_loss = RPNLoss(shape, downsampled_rate, [scale], RPN_ANCHOR_RATIOS)(p, gt_positions)
            cls_loss.backward()
            bbox_loss.backward()
        optimizer.step()
