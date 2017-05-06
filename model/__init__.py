import chainer
import chainer.functions as F
import chainer.links as L

class Stage1(chainer.Chain):

    def __init__(self, n_point):
        super(Stage1, self).__init__(
            conv1=L.Convolution2D(3, 128, ksize=9, stride=1, pad=4),
            conv2=L.Convolution2D(128, 128, ksize=9, stride=1, pad=4),
            conv3=L.Convolution2D(128, 128, ksize=9, stride=1, pad=4),
            conv4=L.Convolution2D(128, 32, ksize=5, stride=1, pad=2),
            conv5=L.Convolution2D(32, 512, ksize=9, stride=1, pad=4),
            conv6=L.Convolution2D(512, 512, ksize=1, stride=1, pad=0),
            conv7=L.Convolution2D(512, n_point+1, ksize=1, stride=1, pad=0),
        )

    def __call__(self, x):
        h = self.conv1(x)
        h = F.relu(h)
        h = F.max_pooling_2d(h, ksize=3, stride=2)
        h = self.conv2(h)
        h = F.relu(h)
        h = F.max_pooling_2d(h, ksize=3, stride=2)
        h = self.conv3(h)
        h = F.relu(h)
        h = F.max_pooling_2d(h, ksize=3, stride=2)
        h = self.conv4(h)
        h = F.relu(h)
        h = self.conv5(h)
        h = F.relu(h)
        h = self.conv6(h)
        h = F.relu(h)
        h = self.conv7(h)

        return h

class Branch(chainer.Chain):

    def __init__(self):
        super(Branch, self).__init__(
            conv1=L.Convolution2D(3, 128, ksize=9, stride=1, pad=4),
            conv2=L.Convolution2D(128, 128, ksize=9, stride=1, pad=4),
            conv3=L.Convolution2D(128, 128, ksize=9, stride=1, pad=4),
        )

    def __call__(self, x):
        h = self.conv1(x)
        h = F.relu(h)
        h = F.max_pooling_2d(h, ksize=3, stride=2)
        h = self.conv2(h)
        h = F.relu(h)
        h = F.max_pooling_2d(h, ksize=3, stride=2)
        h = self.conv3(h)
        h = F.relu(h)
        h = F.max_pooling_2d(h, ksize=3, stride=2)

        return h

class StageN(chainer.Chain):

    def __init__(self, n_point):
        super(StageN, self).__init__(
            conv0=L.Convolution2D(128, 32, ksize=5, stride=1, pad=2),
            conv1=L.Convolution2D(32 + (n_point+1) + 1, 128, ksize=11, stride=1, pad=5),
            conv2=L.Convolution2D(128, 128, ksize=11, stride=1, pad=5),
            conv3=L.Convolution2D(128, 128, ksize=11, stride=1, pad=5),
            conv4=L.Convolution2D(128, 128, ksize=1, stride=1, pad=0),
            conv5=L.Convolution2D(128, n_point+1, ksize=1, stride=1, pad=0),
        )

    def __call__(self, pmap, fmap, cmap):
        fmap = self.conv0(fmap)
        fmap = F.relu(fmap)
        cmap = F.average_pooling_2d(cmap, ksize=8, stride=8)

        h = F.concat((fmap, pmap, cmap), 1)
        h = self.conv1(h)
        h = F.relu(h)
        h = self.conv2(h)
        h = F.relu(h)
        h = self.conv3(h)
        h = F.relu(h)
        h = self.conv4(h)
        h = F.relu(h)
        h = self.conv5(h)

        return h

class CPM(chainer.Chain):

    def __init__(self, n_point, n_stage):
        super(CPM, self).__init__()
        self.add_link('branch', Branch())
        self.add_link('stage1', Stage1(n_point))
        links = []
        for i in xrange(n_stage-1):
            links += [('stage{}'.format(i+2), StageN(n_point))]
        for l in links:
            self.add_link(*l)

        self.forward = links
        self.train = True

    def clear(self):
        self.loss = None

    def __call__(self, image, cmap, t):
        self.clear()
        h1 = self.stage1(image)
        h2 = self.branch(image)
        self.loss = F.mean_squared_error(h1, t)

        for name, _ in self.forward:
            f = getattr(self, name)
            h1 = f(h1, h2, cmap, train=self.train)
            self.loss += F.mean_squared_error(h1, t)

        if self.train:
            return h1, self.loss
        else:
            return h1
