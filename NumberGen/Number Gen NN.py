import mnist_loader as mnl
import NeuralNetwork as network
import numpy as np
from PIL import Image


trn = input('Train the network (y/n)\n: ').upper()

if trn == 'Y':
    training_data, validation_data, test_data = mnl.load_data_wrapper()
    imgen = network.Network([10, 30, 50, 50, 784])

    imdata = []
    for i in tuple(training_data):
        imdata.append((i[1], i[0]))

    print('\n--data loaded--\n')
    imgen.SGD(imdata, 10, 10, 0.3)
    print('\n--Training Complete--\n')
    imgen.save('data/imgen')

else:
    num = input('enter a number\n: ')
    imgen = network.Network([])
    imgen.load('data/imgen')
    pout = [[]for i in range(28)]
    for n in range(len(num)):

        data = [[0] for n in range(0,10)]
        data[int(str(num)[n])] = [1]
        pix = imgen.feedforward(data)
        #pix = imgen.feedforward([[1],[1],[1],[1],[1],[1],[1],[1],[1],[1]])

        _pout = [[1 for i in range(28)] for i in range(28)]

        for r in range(len(pix)):
            pix[r] = -255*(pix[r]-1)

        for i in range(28):
            for j in range(28):
                _pout[i][j] = float(pix[j+i*28])
            pout[i] += _pout[i]


    img = Image.fromarray(np.uint8(pout))
    img.show()
    #img.save('data/out.bmp')
