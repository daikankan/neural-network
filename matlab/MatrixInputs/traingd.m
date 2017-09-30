function [ net ] = traingd( netinit, P, T, iteration, learnrate, momentum )
%TRAINGD Summary of this function goes here
%   Detailed explanation goes here

net = netinit;

nop = size(P, 3); % number of patterns

c11 = zeros(size(net.weights{1, 1}));
c12 = zeros(size(net.weights{1, 2}));
c21 = zeros(size(net.weights{2, 1}));
c22 = zeros(size(net.weights{2, 2}));

for iter = 1:iteration
    
    for j = 1:nop
        
        [ grads ] = calcg( net, P(:, :, j), T(:, j) );
        
        c11 = momentum * c11 - (1 - momentum) * learnrate * grads{1, 1};
        net.weights{1, 1} = net.weights{1, 1} + c11; % update input layer weights
        
        c12 = momentum * c12 - (1 - momentum) * learnrate * grads{1, 2};
        net.weights{1, 2} = net.weights{1, 2} + c12; % update input layer biases
        
        c21 = momentum * c21 - (1 - momentum) * learnrate * grads{2, 1};
        net.weights{2, 1} = net.weights{2, 1} + c21; % update output layer weights
        
        c22 = momentum * c22 - (1 - momentum) * learnrate * grads{2, 2};
        net.weights{2, 2} = net.weights{2, 2} + c22; % update output layer biases
        
    end
end
end

