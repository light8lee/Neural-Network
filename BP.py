import math
from numpy import *

def sigmoid(x):
    return 1.0 / (1.0 + math.exp(-x))

def dsigmoid(x):
    return sigmoid(x) * (1.0 - sigmoid(x))

class BP(object):
    def __init__(self, i, j, k):
        self.Weight_ji = matrix(random.rand(j, i) * 2 - 1)
        self.Weight_kj = matrix(random.rand(k, j) * 2 - 1)
        self.bias = [random.rand(j), random.rand(k)]
        self.sigmoid = vectorize(sigmoid)
        #self.n = None
        self.e = None
        self.H_j = None
        self.O_k = None
        print 'Weight_ji----------------'
        print self.Weight_ji
        print 'Weight_kj----------------'
        print self.Weight_kj
    
    def run_forward_feed(self, inputs):
        X = matrix(inputs).T
        self.H_j = self.sigmoid(self.Weight_ji * X + matrix(self.bias[0]).T)
        self.O_k = self.sigmoid(self.Weight_kj * self.H_j + matrix(self.bias[1]).T)
        #return self.O_k
    
    def train(self, n, inputs, targets):
        self.run_forward_feed(inputs)
        self.e = matrix(targets).T - self.O_k
        self.Weight_ji = self.Weight_ji + n * self.H_j * (1-self.H_j).T * self.Weight_kj.T * self.e * matrix(inputs)
        self.Weight_kj = self.Weight_kj + n * self.e * self.H_j.T
        print 'Weight_ji----------------'
        print self.Weight_ji
        print 'Weight_kj----------------'
        print self.Weight_kj
        #TODO adjust the bias weight
        pass

if __name__ == '__main__':
    A = BP(3, 4, 2)
    In = [4, 3, 6]
    Tar = [5, 8]
    A.train(0.3, In, Tar)

