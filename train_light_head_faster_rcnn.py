from mxnet import gluon
from mxnet.gluon import nn

from src.dataset import TianchiOCRDataset, TianchiOCRDataLoader

N_MAX_EPOCHS = 10
dataset = TianchiOCRDataset('/Users/rlan/datasets/ICPR/train_1000/image_1000', '/Users/rlan/datasets/ICPR/train_1000/txt_1000')
loader = TianchiOCRDataLoader(dataset, shuffle=False)

for epoch in range(N_MAX_EPOCHS):
    for im, label in loader:
        pass



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
