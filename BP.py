import math
from numpy import *

def sigmoid(x):
    return 1.0 / (1.0 + math.exp(-x))

def dsigmoid(x):
    return sigmoid(x) * (1.0 - sigmoid(x))

class BP(object):

    @staticmethod
    def get_error(errs):
        return sum(errs ** 2) / 2

    def __init__(self, i, j, k):
        self.Weight_ji = matrix(random.rand(j, i) * 2 - 1)
        self.Weight_kj = matrix(random.rand(k, j) * 2 - 1)
        self.bias = [matrix(random.rand(j)).T, matrix(random.rand(k)).T]
        self.sigmoid = vectorize(sigmoid)
        self.n = None
        self.inputs = None
        self.e = None
        self.H_j = None
        self.O_k = None
        #print 'Weight_ji----------------'
        #print self.Weight_ji
        #print 'Weight_kj----------------'
        #print self.Weight_kj
        #print 'bias---------------------'
        #print self.bias

    def run_forward_feed(self):
        X = self.inputs
        #print 'Weight_ji----------------'
        #print self.Weight_ji
        #print 'X------------------------'
        #print X
        #print 'bias[0]------------------'
        #print self.bias[0]

        self.H_j = self.sigmoid(self.Weight_ji * X + self.bias[0])
        self.O_k = self.sigmoid(self.Weight_kj * self.H_j + self.bias[1])
        return self.O_k

    def run_back_propagate(self):
        W2 = self.Weight_kj #the copy of the old Weight_kj
        n = self.n
        self.Weight_ji = self.Weight_ji + n * self.H_j * (1-self.H_j).T * W2.T * self.e * self.inputs.T
        self.Weight_kj = W2 + n * self.e * self.H_j.T
        self.bias[0] = self.bias[0] + n * self.H_j * (1-self.H_j).T * W2.T * self.e
        self.bias[1] = self.bias[1] + n * self.e

    def train(self, n, tests, max_iterations=1000):
        #self.inputs = matrix(inputs).T
        self.n = n
        for i in range(max_iterations):
            for p in tests:
                self.inputs = matrix(p[0]).T
                self.run_forward_feed()
                self.e = matrix(p[1]).T - self.O_k
                self.run_back_propagate()
                if i % 50 == 0:
                    print 'error is', BP.get_error(self.e.getA())

        for p in tests:
            self.test(p[0])

    def test(self, inputs):
        self.inputs = matrix(inputs).T
        print '(', inputs, ") -> ", self.run_forward_feed()

if __name__ == '__main__':
    A = BP(2, 4, 1)
    In = [4, 3, 6]
    pat = [
            [[0, 0], [1]],
            [[0, 1], [1]],
            [[1, 0], [1]],
            [[1, 1], [0]]
            ]
    A.train(0.3, pat)

