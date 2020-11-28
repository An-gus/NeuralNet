import random
import numpy as np
import pickle


def sigmoid(x):
    return 1/(1+np.exp(-x))


def sigmoid_der(x):
    return sigmoid(x)*(1-sigmoid(x))


class Network(object):

    def __init__(self, sizes):
        self.num_layers = len(sizes)
        self.sizes = sizes
        self.biases = [np.random.randn(y, 1) for y in sizes[1:]]
        self.weights = [np.random.randn(y,x) for x, y in zip(sizes[:-1], sizes[1:])]

    def feedforward(self,a):
        for b, w in zip(self.biases, self.weights):
            a = sigmoid(np.dot(w,a)+b)
        return a

    def SGD(self, data, epochs, bsize, lr, tst_data=None):

        if tst_data:
            test_data = list(tst_data)
            n_test = len(test_data)
        data = list(data)
        for j in range(epochs):
            n = len(data)
            random.shuffle(data)
            batches = [data[k:k+bsize] for k in range(0,n,bsize)]
            for batch in batches:
                self.update(batch,lr)

            if tst_data:
                print("Epoch {0}: {1} / {2}".format(j, self.test(test_data), n_test))
            else:
                print('Epoch {0} complete'.format(j))

    def update(self,batch,lr):
        grad_b = [np.zeros(b.shape) for b in self.biases]
        grad_w = [np.zeros(w.shape) for w in self.weights]
        for x,y in batch:
            delta_grad_b, delta_grad_w = self.backprop(x, y)
            grad_b = [nb + dnb for nb, dnb in zip(grad_b, delta_grad_b)]
            grad_w = [nw + dnw for nw, dnw in zip(grad_w, delta_grad_w)]
            self.weights = [w-(lr/len(batch))*nw for w, nw in zip(self.weights, grad_w)]
            self.biases = [b-(lr/len(batch))*nb for b, nb in zip(self.biases, grad_b)]

    def backprop(self, x, y):
        grad_b = [np.zeros(b.shape) for b in self.biases]
        grad_w = [np.zeros(w.shape) for w in self.weights]
        activation = x
        activations = [x]
        zs = []
        for b, w in zip(self.biases, self.weights):
            z = np.dot(w, activation) + b
            zs.append(z)
            activation = sigmoid(z)
            activations.append(activation)
        delta = self.cost_der(activations[-1], y) * sigmoid_der(zs[-1])
        grad_b[-1] = delta
        grad_w[-1] = np.dot(delta, activations[-2].transpose())
        for l in range(2, self.num_layers):
            z = zs[-l]
            sp = sigmoid_der(z)
            delta = np.dot(self.weights[-l + 1].transpose(), delta) * sp
            grad_b[-l] = delta
            grad_w[-l] = np.dot(delta, activations[-l - 1].transpose())
        return (grad_b, grad_w)

    def cost_der(self,o,y):
        return o-y

    def test(self,test_data):
        test_results = [(np.argmax(self.feedforward(x)), y)
                        for (x, y) in test_data]
        return sum(int(x == y) for (x, y) in test_results)

    def load(self, file):
        [self.num_layers, self.sizes, self.weights, self.biases] = pickle.load(open(file, 'rb'))

    def save(self, file):
        with open(file, 'wb') as f:
            pickle.dump([self.num_layers, self.sizes, self.weights, self.biases], f)
            f.close()
