#!/usr/bin/python2

from numpy import *
import numpy as np

class MLP:
    def __init__(self, layerSize = []):
        self.numLayer = len(layerSize)  # number of layers
        assert self.numLayer >= 3  # input layer : hidden layer : output layer
        self.layerSize = layerSize  # each layer's size
        self.netParams = []  # learnable parameters
        self.netParams_diff = []  # diff of learnable parameters
        self.netParams_diff_prev = []  # for momentum learning
        self.nodes = []  # intermediate outputs
        self.nodes_diff = []  #  diff of intermediate outputs
        for i in range(self.numLayer-1):
            layerParams = []
            layerParams_diff = []
            layerParams_diff_prev = []
            # weights of each layer
            layerParams.append(np.random.rand(self.layerSize[i+1],
                                              self.layerSize[i]) - 0.5)
            layerParams_diff.append(np.zeros([self.layerSize[i+1],
                                              self.layerSize[i]]))
            layerParams_diff_prev.append(np.zeros([self.layerSize[i+1],
                                                   self.layerSize[i]]))
            # bias of each layer
            layerParams.append(np.random.rand(self.layerSize[i+1],1) - 0.5)
            layerParams_diff.append(np.zeros([self.layerSize[i+1],1]))
            layerParams_diff_prev.append(np.zeros([self.layerSize[i+1],1]))
            self.netParams.append(layerParams)
            self.netParams_diff.append(layerParams_diff)
            self.netParams_diff_prev.append(layerParams_diff_prev)
        for i in range(self.numLayer):
            self.nodes.append(np.zeros([self.layerSize[i], 1]))
            self.nodes_diff.append(np.zeros([self.layerSize[i], 1]))

                                             
    def forward(self, input):
        assert input.shape[0] == self.layerSize[0] 
        self.nodes[0] = input
        for i in range(self.numLayer-2):
            wx = np.dot(self.netParams[i][0], self.nodes[i])  # inner product
            wxb = wx + self.netParams[i][1]  # add bias
            self.nodes[i+1] = 1 / (1 + np.exp(-wxb))  # sigmoid function
        wx = np.dot(self.netParams[self.numLayer-2][0], self.nodes[self.numLayer-2])
        wxb = wx + self.netParams[self.numLayer-2][1]
        self.nodes[self.numLayer-1] = wxb
        return self.nodes[self.numLayer-1]  # return the output of the net
            
                                             
    def cleardiff(self):
        for i in range(self.numLayer-1):
            self.netParams_diff[i][0] = np.zeros([self.layerSize[i+1],
                                                  self.layerSize[i]])
            self.netParams_diff[i][1] = np.zeros([self.layerSize[i+1], 1])

                                             
    def calcdiff(self, inputs, targets, batchSize = 1):
        MLP.cleardiff(self)
        # input dim eq input layersize
        assert inputs.shape[0] == self.layerSize[0]
        # number of examples eq batchSize
        assert batchSize == inputs.shape[1]
        for i in range(batchSize):
            input = inputs.take([i], axis=1)
            target = targets.take([i], axis=1)
            MLP.forward(self, input)

            self.nodes_diff[self.numLayer-1] = (
                self.nodes[self.numLayer-1] - target)
            self.netParams_diff[self.numLayer-2][0] += np.dot(
                self.nodes_diff[self.numLayer-1], self.nodes[self.numLayer-2].T)
            self.netParams_diff[self.numLayer-2][1] += (
                self.nodes_diff[self.numLayer-1])

            for j in range(self.numLayer-3, -1, -1):
                activ_diff = self.nodes[j+1] * (1 - self.nodes[j+1])
                self.nodes_diff[j+1] = activ_diff * (
                    (np.dot(self.netParams[j+1][0].T, self.nodes_diff[j+2])))
                self.netParams_diff[j][0] += np.dot(
                    self.nodes_diff[j+1], self.nodes[j].T)
                self.netParams_diff[j][1] += self.nodes_diff[j+1]

        # mean value of gradient
        for i in range(self.numLayer-1):
            self.netParams_diff[i][0] /= batchSize
            self.netParams_diff[i][1] /= batchSize

                                             
    def update(self, learnRate = 0.01, momentum = 0):
        for i in range(self.numLayer-1):
            self.netParams_diff_prev[i][0] *= momentum
            self.netParams_diff_prev[i][0] -= (1-momentum) * learnRate * (
                                              self.netParams_diff[i][0])
            self.netParams[i][0] += self.netParams_diff_prev[i][0]
                                             
            self.netParams_diff_prev[i][1] *= momentum
            self.netParams_diff_prev[i][1] -= (1-momentum) * learnRate * (
                                              self.netParams_diff[i][1])
            self.netParams[i][1] += self.netParams_diff_prev[i][1]
