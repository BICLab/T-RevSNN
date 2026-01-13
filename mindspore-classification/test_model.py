import mindspore as ms
from mindspore import Tensor
import numpy as np
from mindspore.ops import composite as C


ms.set_context(
    mode=ms.PYNATIVE_MODE,
    device_target="CPU"
)

from models.cifar.ms_spike_rev_reuse import tiny


model = tiny(True, num_classes=10)
model.set_train()

loss_fn = ms.nn.SoftmaxCrossEntropyWithLogits(sparse=True)

x = Tensor(np.random.rand(2, 3, 224, 224).astype(np.float32))
y = Tensor(np.random.randint(0, 10, (2,)), ms.int32)

out = model(x)
for i in out:
    print(i.shape)

grad_fn = C.GradOperation(get_all=False)

grad_fn(model)(x)
