import time

from mxnet import gluon, nd, autograd, metric, init, gpu

from src.dataset import get_iterators, training_targets
from src.ssd import ToySSD
from src.loss import FocalLoss, SmoothL1Loss

root_url = ('https://apache-mxnet.s3-accelerate.amazonaws.com/'
            'gluon/dataset/pikachu/')
data_dir = './data/pikachu/'
dataset = {'train.rec': 'e6bcb6ffba1ac04ff8a9b1115e650af56ee969c8',
           'train.idx': 'dcf7318b2602c06428b9988470c731621716c393',
           'val.rec': 'd6c33f799b4d058e82f2cb5bd9a976f69d72d520'}
# for k, v in dataset.items():
#     gluon.utils.download(root_url+k, data_dir+k, sha1_hash=v)

data_shape = 256
batch_size = 32
rgb_mean = nd.array([123, 117, 104])

train_data, test_data, class_names, num_class = get_iterators(data_dir, data_shape, batch_size)

cls_metric = metric.Accuracy()
box_metric = metric.MAE()
cls_loss = FocalLoss()
box_loss = SmoothL1Loss()

ctx = gpu(0)
# the CUDA implementation requres each image has at least 3 lables.
# Padd two -1 labels for each instance
train_data.reshape(label_shape=(3, 5))
train_data = test_data.sync_label_shape(train_data)

net = ToySSD(num_class)
net.initialize(init.Xavier(magnitude=2), ctx=ctx)
trainer = gluon.Trainer(net.collect_params(),
                        'sgd', {'learning_rate': 0.1, 'wd': 5e-4})

for epoch in range(30):
    # reset data iterators and metrics
    train_data.reset()
    cls_metric.reset()
    box_metric.reset()
    tic = time.time()
    for i, batch in enumerate(train_data):
        x = batch.data[0].as_in_context(ctx)
        y = batch.label[0].as_in_context(ctx)
        with autograd.record():
            anchors, class_preds, box_preds = net(x)
            box_target, box_mask, cls_target = training_targets(
                anchors, class_preds, y)
            # losses
            loss1 = cls_loss(class_preds, cls_target)
            loss2 = box_loss(box_preds, box_target, box_mask)
            loss = loss1 + loss2
        loss.backward()
        trainer.step(batch_size)
        # update metrics
        cls_metric.update([cls_target], [class_preds.transpose((0, 2, 1))])
        box_metric.update([box_target], [box_preds * box_mask])

    print('Epoch %2d, train %s %.2f, %s %.5f, time %.1f sec' % (
        epoch, *cls_metric.get(), *box_metric.get(), time.time() - tic
    ))
