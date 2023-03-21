import paddle

import paddle.vision.transforms as T
# transform = T.Normalize(mean=[127.5], std=[127.5], data_format='CHW')
#
# # 下载数据集
# train_dataset = paddle.vision.datasets.MNIST(mode='train', transform=transform)
# val_dataset = paddle.vision.datasets.MNIST(mode='test', transform=transform)
#
#
# mnist = paddle.nn.Sequential(
#     paddle.nn.Flatten(),
#     paddle.nn.Linear(784, 512),
#     paddle.nn.ReLU(),
#     paddle.nn.Dropout(0.2),
#     paddle.nn.Linear(512, 10)
# )
#
# # 预计模型结构生成模型对象，便于进行后续的配置、训练和验证
# model = paddle.Model(mnist)
#
# # 模型训练相关配置，准备损失计算方法，优化器和精度计算方法
# model.prepare(paddle.optimizer.Adam(parameters=model.parameters()),
#                 paddle.nn.CrossEntropyLoss(),
#                 paddle.metric.Accuracy())
#
# # 开始模型训练
# model.fit(train_dataset,
#             epochs=5,
#             batch_size=64,
#             verbose=1)
#
# model.evaluate(val_dataset, verbose=0)

print("gpu" in paddle.device.get_device())
paddle.device.set_device('cpu')
print(paddle.device.get_device())
