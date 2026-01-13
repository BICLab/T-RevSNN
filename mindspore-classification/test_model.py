import mindspore as ms
from mindspore import Tensor
import numpy as np


ms.set_context(
    mode=ms.GRAPH_MODE,
    device_target="CPU"
)

from models.cifar.ms_spike_rev_reuse import tiny


model = tiny(True, num_classes=10)
model.set_train()

x = Tensor(np.random.rand(1, 3, 224, 224).astype(np.float32))
out = model(x)
for i in out:
    print(i.shape)
