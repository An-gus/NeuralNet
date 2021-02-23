import math, requests
import numpy as np
import matplotlib.dates as md
import datetime
from sklearn.preprocessing import MinMaxScaler
import matplotlib.pyplot as plt
from keras.models import Sequential
from keras.layers import Dense, LSTM


def predict(data, model):
    inp = np.array(data)
    inp = np.reshape(inp, (inp.shape[0], inp.shape[1], 1))
    pred = model.predict(inp)
    return pred


def genData(length, data, model, scaler, maxTime):
    out = []
    time = []
    for i in range(length):
        nprice = predict(data, model)

        ndata = data[-1]
        ndata = np.delete(ndata, 0, 0)
        ndata = np.append(ndata, [nprice[-1]], 0)
        data = np.delete(data, 0, 0)
        data = np.append(data, [ndata], 0)

        nprice = scaler.inverse_transform(nprice)
        out.append(nprice[-1])
        time.append(maxTime + datetime.timedelta(days=(i+1)))
    return out, time


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

    return time, prices, scope, futureDays


def train(prices, bs, e, inpLength=40):

    tdl = math.ceil(len(prices) * 0.8)
    data = np.array([prices]).reshape(len(prices), 1)
    scl = MinMaxScaler(feature_range=(0, 1))
    sclData = scl.fit_transform(data)
    tdata = sclData[0:tdl]

    xlen = inpLength
    xtrain, ytrain = [], []
    for i in range(xlen, tdl):
        xtrain.append(tdata[i - xlen:i])
        ytrain.append(tdata[i])

    xtrain, ytrain = np.array(xtrain), np.array(ytrain)
    xtrain = np.reshape(xtrain, (tdl - xlen, xlen, 1))

    model = Sequential()
    model.add(LSTM(50, return_sequences=True, input_shape=(xlen, 1)))
    model.add(LSTM(50, return_sequences=False))
    model.add(Dense(25))
    model.add(Dense(1))

    model.compile(optimizer='adam', loss='mean_squared_error')

    model.fit(xtrain, ytrain, batch_size=bs, epochs=e)

    #--Test AI--#

    # testdata = sclData[tdl-xlen:]
    # xtest = []
    # ytest = data[tdl:]
    # for i in range(xlen, len(testdata)):
    #     xtest.append(testdata[i-xlen:i])
    # xtest = np.array(xtest)
    # xtest = np.reshape(xtest, (xtest.shape[0], xtest.shape[1], 1))
    # pred = model.predict(xtest)
    # pred = scl.inverse_transform(pred)


    return model, scl


if __name__ == '__main__':

    time, prices, scope, ftrd = getData()

    setLength = 20
    bot, scaler = train(prices, 5, 50, setLength)

    envals = np.array(prices[-setLength:]).reshape(setLength, 1)
    envals = scaler.transform(envals)
    envals = np.reshape(envals, (1, setLength, 1))

    pred, extraTime = genData(ftrd, envals, bot, scaler, time[-1])
    pred.insert(0, prices[-1])
    extraTime.insert(0, time[-1])

    plt.style.use('fivethirtyeight')
    plt.title(scope.title()+' Price')
    plt.plot(time, prices)
    plt.xlabel('Date', fontsize=18)
    plt.ylabel('Price', fontsize=18)
    # plt.plot(time[-len(validation):], validation)
    plt.plot(extraTime, pred)
    plt.legend(['Actual', 'Prediction'])
    plt.show()
