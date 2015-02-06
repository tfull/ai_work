# coding: utf-8

import random as R
import sys

def function(ws, xs):
    return sign_p(sum([w * x for (w, x) in zip(ws, xs)]))

def sign_p(x):
    if x >= 0:
        return 1
    else:
        return -1

class Perceptron:
    def __init__(self, s_n, a_n, r_n, eta):
        self.s_n = s_n
        self.r_n = r_n
        self.a_n = a_n
        self.eta = eta
        self.sa_m = [[R.random() for j in range(self.a_n)] for i in range(self.s_n)]
        self.ar_m = [[1 / self.r_n for j in range(self.r_n)] for i in range(self.a_n)]

    def train(self, train_l):
        iterator = 0
        for tr in train_l:
            iterator += 1
            (xs, ts) = tr
            (os, ys) = self.connect(xs)

            for i in range(self.r_n):
                for j in range(self.a_n):
                    self.ar_m[j][i] += self.eta * (ts[i] - os[i]) * ys[j]

    def connect(self, xs):
        ys = [0 for i in range(self.a_n)]

        for a_i in range(self.a_n):
            ys[a_i] = function([self.sa_m[i][a_i] for i in range(self.s_n)], xs)

        os = [0 for i in range(self.r_n)]

        for r_i in range(self.r_n):
            os[r_i] = function([self.ar_m[i][r_i] for i in range(self.a_n)], ys)

        return (os, ys)

    def execute(self, xs):
        return self.connect(xs)[0]
