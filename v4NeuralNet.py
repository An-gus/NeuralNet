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
        self.weights = [np.random.randn(y, x)/(y**0.5) for x, y in zip(self.sizes[:-1], self.sizes[1:])]

    def feedforward(self, a):
        for b, w in zip(self.biases, self.weights):
            a = sig(np.dot(w, a)+b)
        return a

    def train(self, inp, lr, epochs, bs, cp=False):
        print('[*]Beginning Training[*]')
        for epoch in range(epochs):
            if cp and epoch % (epochs/10) == 0 and epoch != 0:
                self.save('checkPoint.v4Net.AI')
            print('    epoch {0} of {1}'.format(epoch+1, epochs))
            n = len(inp)
            random.shuffle(inp)
            batches = [inp[k:k + bs] for k in range(0, n, bs)]
            for batch in batches:
                DW = [np.zeros(w.shape) for w in self.weights]
                DB = [np.zeros(b.shape) for b in self.biases]
                for data in batch:
                    _W = [np.zeros(w.shape) for w in self.weights]
                    _B = [np.zeros(b.shape) for b in self.biases]
                    _a = data[0]
                    a = [_a]
                    os = []
                    for b, w in zip(self.biases, self.weights):
                        o = np.dot(w, _a) + b
                        os.append(o)
                        _a = sig(o)
                        a.append(_a)

                    # gd #
                    dedb = (a[-1] - data[1]) * (a[-1]*(1-a[-1]))
                    _B[-1] = dedb
                    _W[-1] = np.dot(dedb, a[-2].transpose())
                    for n in range(2, self.nlays):
                        o = os[-n]
                        dodi = sig(o)*(1-sig(o))
                        dedb = np.dot(self.weights[-n+1].transpose(), dedb) * dodi
                        _B[-n] = dedb
                        _W[-n] = np.dot(dedb, a[-n-1].transpose())

                    DB = [gb + dgb for gb, dgb in zip(DB, _B)]
                    DW = [gw + dgw for gw, dgw in zip(DW, _W)]
                    self.weights = [w - (lr / len(batch)) * gw for w, gw in zip(self.weights, DW)]
                    self.biases = [b - (lr / len(batch)) * gb for b, gb in zip(self.biases, DB)]
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
        net.train(i, 0.005, 50, 10)
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
        img.save('v4Net.bmp')
        if input('\nWould you like to save the network?[Y/N]: ').upper() == 'Y':
            net.save('v4Net.AI')
    else:
        net.load('v4Net.AI')
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
        # img.save('v4Net.bmp')
