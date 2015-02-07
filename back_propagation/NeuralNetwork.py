# coding: utf-8

import math
import random as R

def sigmoid(x):
    return 1 / (1 + math.exp(- x))

class NeuralNetwork:
    def __init__(self, s_n, a_n, r_n, theta, eta):
        self.s_n = s_n
        self.a_n = a_n
        self.r_n = r_n
        self.theta = theta
        self.eta = eta
        self.sa_m = [[R.random() for j in range(a_n)] for i in range(s_n)]
        self.ar_m = [[R.random() for j in range(r_n)] for i in range(a_n)]
# self.wa = [[- R.random() for i in range(a_n)]]
#self.wr = [[- R.random() for i in range(r_n)]]

    def function(self, ws, xs):
        return sigmoid(sum([w * x for (w, x) in zip(ws, xs)]) - self.theta)

    def train(self, train_l):

        def delta(ts, os, k):
            return (ts[k] - os[k]) * os[k] * (1 - os[k])

        for tr in train_l:
            (xs, ts) = tr
            (os, hs) = self.connect(xs)

            for r_i in range(self.r_n):
                for a_i in range(self.a_n):
                    self.ar_m[a_i][r_i] += self.eta * delta(ts, os, r_i) * hs[a_i]

            for a_i in range(self.a_n):
                for s_i in range(self.s_n):
                    self.sa_m[s_i][a_i] += self.eta * hs[a_i] * (1 - hs[a_i]) * sum([self.ar_m[a_i][r_i] * delta(ts, os, r_i) for r_i in range(self.r_n)]) * xs[s_i]

    def connect(self, xs):
        hs = [0 for i in range(self.a_n)]
        os = [0 for i in range(self.r_n)]

        for a_i in range(self.a_n):
            hs[a_i] = self.function([self.sa_m[i][a_i] for i in range(self.s_n)], xs)
        for r_i in range(self.r_n):
            os[r_i] = self.function([self.ar_m[i][r_i] for i in range(self.a_n)], hs)

        return (os, hs)

    def execute(self, xs):
        return self.connect(xs)[0]
