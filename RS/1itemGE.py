import NetClass, requests, datetime, os
import numpy as np
import matplotlib.pyplot as plt
from sklearn.preprocessing import MinMaxScaler


def existFile(filename, search_path):

    for root, dir, files in os.walk(search_path):
        if filename in files:
            return True
    return False


def getData():

    itemID = input('Enter the id of the item <default logs> : ')
    if not itemID: itemID = '1511'
    url = 'http://services.runescape.com/m=itemdb_rs/api/graph/{}.json'.format(itemID)
    res = requests.get(url, timeout=10)

    scope = input('Either [d]aily or [a]verage: ')
    if scope == 'd':
        scope = 'daily'
    else:
        scope = 'average'

    prices = []
    time = []
    for point in res.json()[scope]:
        time.append(datetime.datetime.fromtimestamp(int(int(point) / 1000)))
        prices.append(res.json()[scope][point])

    futureDays = input('How many days into the future would you like to predict <Default 30>: ')
    if not futureDays: futureDays = 30
    else: futureDays = int(futureDays)

    return time, prices, scope, futureDays, itemID


def prepData(data, l):
    sData = []
    for i in range(l, len(data)):
        x = np.array(data[i-l:i]).reshape(l, 1)
        y = np.array(data[i]).reshape(1, 1)
        sData.append((x, y))
    return sData


def plotData(x, y, x1, y1, scope):

    plt.style.use('fivethirtyeight')
    plt.title(scope.title() + ' Price')
    plt.xlabel('Date', fontsize=18)
    plt.ylabel('Price', fontsize=18)

    plt.plot(x, y)
    plt.plot(x1, y1)

    plt.legend(['Actual', 'Prediction'])
    plt.show()


def predict(network, x, num, t):
    time = []
    out = []
    for i in range(num):
        inp = np.array(x).reshape(len(x), 1)
        inp = inp.astype('float')
        y = network.feedforward(inp)
        out.append(y[0])
        #x.pop(0)
        #x.append(y[0])
        x = np.delete(x, 0, 0)
        x = np.append(x, y, 0)
        time.append(t + datetime.timedelta(days=(i + 1)))

    return out, time


def option():
    print('\nPick an option:')
    print('[1]->Predict from web')
    print('[2]->Predict from file')
    print('[3]->Predict from data')
    print('[Q]->Exit')
    return input('> ').upper()


def fromWeb(ID, scope, xlen):
    print('\n[*] Loading Data [*]')
    url = 'http://services.runescape.com/m=itemdb_rs/api/graph/{}.json'.format(ID)
    res = requests.get(url, timeout=10)

    prices = []
    time = []
    for point in res.json()[scope]:
        time.append(datetime.datetime.fromtimestamp(int(int(point) / 1000)))
        prices.append(res.json()[scope][point])
    print('[*] Data Loaded [*]\n')

    ndays = input('How many days to predict <default 30>: ')
    if not ndays: ndays = 30
    else: ndays = int(ndays)

    past = input('Starting how many days in the past 1->'+str(180-xlen)+' <default 1>: ')
    if not past: past = 1
    else: past = int(past)

    if past > (180-xlen) or past < 1:
        print('\n[*] Error: Invalid start date [*]')
        return
    if ndays < 1:
        print('\n[*] Error: Invalid number of days [*]')

    return prices[-(past+xlen):-past], ndays, [-(past+xlen), -past], prices, time


if input('Load a network? [Y/N]: ').upper() == 'Y':

    ID = input('Enter the ID of the item: ')
    scope = input('Enter the scope of the data, [a]verage or [d]aily: ')
    if scope == 'd': scope = 'daily'
    else: scope = 'average'

    if not existFile(ID+scope+'.data', os.path.dirname(os.path.realpath(__file__))+'/data'):
        raise Exception('No existing model for ID='+ID+' and Scope='+scope)

    net = NetClass.Net([1,1,1,1])
    net.load('data/'+ID+scope+'.data')

    print('\n[*] Network Loaded [*]')

    Xlength = net.sizes[0]
    scl = net.scaler

    while True:
        opt = option()
        if opt == '1':
            data, days, timePos, prices, times = fromWeb(ID, scope, Xlength)
            _data = scl.fit_transform(np.array(data).reshape(len(data), 1))

            y, y_t = predict(net, _data, days, times[timePos[1]])

            _y = scl.inverse_transform(np.array(y).reshape(len(y), 1))
            _y = np.insert(_y, 0, prices[timePos[1]])
            y_t.insert(0, times[timePos[1]])

            plotData(times, prices, y_t, _y, scope)
        elif opt == '2':
            pass
        elif opt == '3':
            pass
        elif opt == 'Q':
            break
        else:
            print('\n[*] Error: Invalid input [*]')


else:

    Xlength = 30
    time, _prices, scope, fdays, ID = getData()

    scl = MinMaxScaler(feature_range=(0, 1))
    prices = scl.fit_transform(np.array(_prices).reshape(len(_prices),1))

    x = prepData(prices, Xlength)
    net = NetClass.Net([Xlength, 75, 45, 30, 1], scl)
    net.train(x, 0.05, 500, 1)

    y, ftime = predict(net, prices[-Xlength:], fdays, time[-1])

    y = np.insert(y, 0, prices[-1])
    _y = scl.inverse_transform(np.array(y).reshape(len(y),1))

    ftime.insert(0, time[-1])

    plotData(time, _prices, ftime, _y, scope)

    if input('Would you like to save the network? [Y/N]: ').upper() == 'Y':
        net.save('data/'+ID+scope+'.data')
