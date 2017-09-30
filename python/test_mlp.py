#!/usr/bin/python2

import numpy as np
import matplotlib.pyplot as plt
from mlp import MLP

def esincos(x):
    return np.exp(x) - x * np.sin(x) * np.cos(x)

if __name__ == "__main__":
    x = np.random.rand(1, 100) * 10
    #x = np.linspace(0,10,100)
    y = (np.sin(x) + 1) / 2

    plt.plot(x[0], y[0], "*r")
    plt.show()

   # print x, y

    net = MLP([1, 10, 1])
    
    for iter in range(10000):
        t1 = [net.forward(x.take([i],axis=1))[0][0] for i in range(100)]
        t = [(net.forward(x.take([i],axis=1)) - y.take([i],axis=1))[0][0] for i in range(100)]
       # print(t)
        #print(x[0])
        #print(t)
        if(iter % 2000 == 0):
            plt.plot(x[0], t1, "g.")
            plt.show()
        error = np.sum(np.abs(t))
        print(error)
        for i in range(100):
            #print "x.take([i], axis=1)"
            #print x.take([i], axis=1)
            net.calcdiff(x.take([i], axis=1), y.take([i], axis=1) ,1)
            net.update(0.1, 0.9)
