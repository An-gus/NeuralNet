import numpy as np
import random
import pickle


def sig(n):
    return 1/(1+np.exp(-n))


class Net(object):

    def __init__(self, lays):
        self.nlays = len(lays)
        self.sizes = lays
        self.biases = [np.random.rand(y, 1) for y in self.sizes[1:]]
        self.weights = [np.random.randn(y, x) for x, y in zip(self.sizes[:-1], self.sizes[1:])]

    def feedforward(self, a):
        for b, w in zip(self.biases, self.weights):
            a = sig(np.dot(w, a)+b)
        return a

    def train(self, inp, lr, epochs, bs, cp=False):
        print('[*]Beginning Training[*]')
        for epoch in range(epochs):
            if cp and epoch % (epochs/10) == 0 and epoch != 0:
                self.save('checkPoint.v2Net.AI')
            print('    epoch {0} of {1}'.format(epoch+1, epochs))
            n = len(inp)
            random.shuffle(inp)
            batches = [inp[k:k + bs] for k in range(0, n, bs)]

            for batch in batches:
                _W = [np.zeros(w.shape) for w in self.weights]
                _B = [np.zeros(b.shape) for b in self.biases]
                for data in batch:
                    _a = data[0]
                    a = [_a]
                    for b, w in zip(self.biases, self.weights):
                        _a = sig(np.dot(w, _a) + b)
                        a.append(_a)
                    # gd #
                    for n in range(len(data[1])):
                        dedo = a[-1][n] - data[1][n]
                        for layer in range(1, len(self.weights)+1):
                            for node in range(len(self.weights[layer-1])):
                                out = a[layer][node]
                                dodi = out * (1-out)
                                dedb = dedo*dodi
                                _b = self.biases[layer-1][node]
                                _b -= (lr*dedb)/bs
                                _B[layer-1][node] = _b
                                for w in range(len(self.weights[layer-1][node])):
                                    didw = a[layer-1][w]
                                    dedw = dedo*dodi*didw
                                    _w = self.weights[layer-1][node][w]
                                    _w -= (lr*dedw)/bs
                                    _W[layer-1][node][w] = _w
                self.weights = _W
                self.biases = _B
        print('[*]Training Complete[*]')

    def load(self, file):
        [self.nlays, self.sizes, self.weights, self.biases] = pickle.load(open(file, 'rb'))

    def save(self, file):
        with open(file, 'wb') as f:
            pickle.dump([self.nlays, self.sizes, self.weights, self.biases], f)
            f.close()


if __name__ == '__main__':
    from PIL import Image

    net = Net([2, 5, 5, 1])

    if input('Train the network?[Y/N]: ').upper() == 'Y':
        i = []
        for x_ in range(-200, 200):
            x = x_ / 100
            for y_ in range(0, 150):
                y = y_ / 100
                i1 = np.array([x, y]).reshape(2, 1)
                y2 = np.exp(-(x ** 2))
                if y2 >= y:
                    # point is under the curve
                    i2 = np.array([0]).reshape(1, 1)
                else:
                    # point is above the curve
                    i2 = np.array([1]).reshape(1, 1)
                i.append((i1, i2))
        net.train(i, 0.005, 20, 1)
        print(net.feedforward(np.array([0, 0.5]).reshape(2, 1)))
        pdata = []
        for y_ in range(150, 0, -1):
            y = y_ / 100
            _data = []
            for x_ in range(-200, 200, 2):
                x = x_ / 100
                i = np.array([x, y]).reshape(2, 1)
                p = (1 - net.feedforward(i)[0][0]) * 255
                _data.append(p)
            pdata.append(_data)

        img = Image.fromarray(np.uint8(pdata))
        img.show()
        # img.save('v2Net.bmp')
        if input('\nWould you like to save the network?[Y/N]: ').upper() == 'Y':
            net.save('v2Net.AI')
    else:
        net.load('v2Net.AI')
        pdata = []
        for y_ in range(150, 0, -1):
            y = y_ / 100
            _data = []
            for x_ in range(-200, 200, 2):
                x = x_ / 100
                i = np.array([x, y]).reshape(2, 1)
                p = (1 - net.feedforward(i)[0][0]) * 255
                _data.append(p)
            pdata.append(_data)

        img = Image.fromarray(np.uint8(pdata))
        img.show()
        # deimg.save('v2Net.bmp')


