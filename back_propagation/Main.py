# coding: utf-8

import sys
import math
import random

import NeuralNetwork as NN

TRAIN_N = 10000
DATA_N = 10000

def main(ws):
    n = len(ws)
    train_l = []
    data_l = []

    for t in range(TRAIN_N):
        xs = []
        for i in range(n):
            xs.append(random.randint(0, 200) / 10 - 10)
        train_l.append((xs, [function(ws, xs)]))

    for t in range(DATA_N):
        xs = []
        for i in range(n):
            xs.append(random.randint(0, 200) / 10 - 10)
        data_l.append((xs, [function(ws, xs)]))

    neuron = NN.NeuralNetwork(n, 12, 1, 0.2, 0.1)
    neuron.train(train_l)

    count = 0

    for data in data_l:
        (xs, t) = data
        result = neuron.execute(xs)
        if sign_y(result[0]) == t[0]:
            count += 1

    print(count / DATA_N)

def function(ws, xs):
    return sign_p(ws[0] + sum([w * x for (w, x) in zip(ws[1:], xs)]))

def sign_y(y):
    if y >= 0.5:
        return 1
    else:
        return -1

def sign_p(x):
    if x >= 0:
        return 1
    else:
        return -1

if __name__ == '__main__':
    args = sys.argv
    main([float(w) for w in args[1:]])
