"""densenet"""
import math
import mindspore as ms
from mindspore import nn, ops

__all__ = ['densenet']


class Bottleneck(nn.Cell):
    """Bottleneck"""
    def __init__(self, inplanes, expansion=4, growthRate=12, dropRate=0):
        # super(Bottleneck, self).__init__()
        super().__init__()
        planes = expansion * growthRate
        self.bn1 = nn.BatchNorm2d(inplanes)
        self.conv1 = nn.Conv2d(inplanes, planes, kernel_size=1, has_bias=False)
        self.bn2 = nn.BatchNorm2d(planes)
        self.conv2 = nn.Conv2d(planes, growthRate, kernel_size=3,
                               padding=1, has_bias=False, pad_mode="pad")
        self.relu = nn.ReLU()
        self.dropRate = dropRate

    def construct(self, x):
        out = self.bn1(x)
        out = self.relu(out)
        out = self.conv1(out)
        out = self.bn2(out)
        out = self.relu(out)
        out = self.conv2(out)
        if self.dropRate > 0:
            out = ops.dropout(out, p=self.dropRate, training=self.training)

        out = ops.cat((x, out), axis=1)

        return out


class BasicBlock(nn.Cell):
    """basic block"""
    # def __init__(self, inplanes, expansion=1, growthRate=12, dropRate=0):
    def __init__(self, inplanes, growthRate=12, dropRate=0):
        # super(BasicBlock, self).__init__()
        super().__init__()
        # planes = expansion * growthRate
        self.bn1 = nn.BatchNorm2d(inplanes)
        self.conv1 = nn.Conv2d(inplanes, growthRate, kernel_size=3,
                               padding=1, has_bias=False, pad_mode="pad")
        self.relu = nn.ReLU()
        self.dropRate = dropRate

    def construct(self, x):
        out = self.bn1(x)
        out = self.relu(out)
        out = self.conv1(out)
        if self.dropRate > 0:
            out = ops.dropout(out, p=self.dropRate, training=self.training)

        out = ops.cat((x, out), 1)

        return out


class Transition(nn.Cell):
    """Transition"""
    def __init__(self, inplanes, outplanes):
        # super(Transition, self).__init__()
        super().__init__()
        self.bn1 = nn.BatchNorm2d(inplanes)
        self.conv1 = nn.Conv2d(inplanes, outplanes, kernel_size=1,
                               has_bias=False, pad_mode="pad")
        self.relu = nn.ReLU()

    def construct(self, x):
        out = self.bn1(x)
        out = self.relu(out)
        out = self.conv1(out)
        out = ops.avg_pool2d(out, 2)
        return out


class DenseNet(nn.Cell):
    """DenseNetwork"""

    def __init__(self, depth=22, block=Bottleneck,
                 dropRate=0, num_classes=10, growthRate=12, compressionRate=2):
        # super(DenseNet, self).__init__()
        super().__init__()

        assert (depth - 4) % 3 == 0, 'depth should be 3n+4'
        n = (depth - 4) / 3 if block == BasicBlock else (depth - 4) // 6

        self.growthRate = growthRate
        self.dropRate = dropRate

        # self.inplanes is a global variable used across multiple
        # helper functions
        self.inplanes = growthRate * 2
        self.conv1 = nn.Conv2d(3, self.inplanes, kernel_size=3, padding=1,
                               has_bias=False, pad_mode="pad")
        self.dense1 = self._make_denseblock(block, n)
        self.trans1 = self._make_transition(compressionRate)
        self.dense2 = self._make_denseblock(block, n)
        self.trans2 = self._make_transition(compressionRate)
        self.dense3 = self._make_denseblock(block, n)
        self.bn = nn.BatchNorm2d(self.inplanes)
        self.relu = nn.ReLU()
        # if num_classes == 10:
        ker_size = 30
        self.avgpool = nn.AvgPool2d(ker_size, pad_mode="pad", stride=ker_size)
        self.fc = nn.Dense(self.inplanes, num_classes)

        # Weight initialization
        # for m in self.modules():
        #     if isinstance(m, nn.Conv2d):
        #         n = m.kernel_size[0] * m.kernel_size[1] * m.out_channels
        #         m.weight.data.normal_(0, math.sqrt(2. / n))
        #     elif isinstance(m, nn.BatchNorm2d):
        #         m.weight.data.fill_(1)
        #         m.bias.data.zero_()

    def _make_denseblock(self, block, blocks):
        layers = []
        for _ in range(blocks):
            # Currently we fix the expansion ratio as the default value
            layers.append(block(self.inplanes, growthRate=self.growthRate, dropRate=self.dropRate))
            self.inplanes += self.growthRate

        return nn.SequentialCell(*layers)

    def _make_transition(self, compressionRate):
        inplanes = self.inplanes
        outplanes = int(math.floor(self.inplanes // compressionRate))
        self.inplanes = outplanes
        return Transition(inplanes, outplanes)

    @ms.jit
    def construct(self, x):
        x = self.conv1(x)

        x = self.trans1(self.dense1(x))
        x = self.trans2(self.dense2(x))
        x = self.dense3(x)
        x = self.bn(x)
        x = self.relu(x)

        x = self.avgpool(x)
        x = x.view(x.shape[0], -1)
        x = self.fc(x)

        return x


def densenet(**kwargs):
    """
    Constructs a ResNet model.
    """
    return DenseNet(**kwargs)
