import torch
from torchvision.transforms import functional as F

from src.transform import TianchiOCRDynamicResize, TianchiOCRClip
from src.dataset import TianchiOCRDataset, TianchiOCRDataLoader
from src.fpn_densenet import FPNDenseNet
from src.rpn import RPNLoss

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
rpn_loss = RPNLoss()

for epoch in range(N_MAX_EPOCHS):
    for im, label in loader:
        im = F.to_tensor(im).view(1, 3, *im.size)
        gt_positions = torch.from_numpy(label[0])

        if torch.cuda.is_available():
            im = im.cuda()
            gt_positions = gt_positions.cuda()

        rpn_proposals = net(im)
        l = rpn_loss(rpn_proposals, gt_positions)
        optimizer.zero_grad()
        l.backward()
        optimizer.step()



# net = nn.Sequential()
# with net.name_scope():
#     net.add(
#         nn.Conv2D(channels=20, kernel_size=5, activation='relu'),
#         nn.MaxPool2D(pool_size=2, strides=2),
#         nn.Conv2D(channels=50, kernel_size=3, activation='relu'),
#         nn.MaxPool2D(pool_size=2, strides=2),
#         nn.Flatten(),
#         nn.Dense(128, activation="relu"),
#         nn.Dense(10)
#     )

# iterator = ImageDetIter(1, (3, 600, 600), path_imgrec='data/pikachu/train.rec')
# for data in iterator:
#     pass

# for image in iterator.draw_next(waitKey=None):
#     pass

# # or let draw_next display using cv2 module
# for image in iterator.draw_next(waitKey=0, window_name='disp'):
#     pass
