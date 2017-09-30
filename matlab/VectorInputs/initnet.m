function [ net ] = initnet( netinit )
%initnet is used to inital the weights of the network.
%
% Dai Kankan 2014.

net = netinit;

net.weights{1, 1} = rand(size(net.weights{1, 1})) - 0.5; % input layer weights
net.weights{1, 2} = rand(size(net.weights{1, 2})) - 0.5; % input layer biases

net.weights{2, 1} = rand(size(net.weights{2, 1})) - 0.5; % output layer weights
net.weights{2, 2} = rand(size(net.weights{2, 2})) - 0.5; % output layer biases
end

