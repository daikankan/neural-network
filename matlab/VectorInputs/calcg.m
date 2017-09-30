function [ grads ] = calcg( net, p, t )
%CALCG Summary of this function goes here
%   Detailed explanation goes here

[ node ] = forward( net, p );

grads = {}; % gradients of all weights

delta{2} = node{2} - t; % o x 1

grads{2, 1} = delta{2} * node{1}'; % o x h?  gradients of output layer weights
grads{2, 2} = delta{2}; % o x h? gradients of output layer biases

delta{1} = node{1} .* (ones(size(node{1})) - node{1}) .* (net.weights{2, 1}' * delta{2}); % i x 1

grads{1, 1} = delta{1} * p';  % h? x i gradients of input layer weights
grads{1, 2} = sum(delta{1}, 2); % h? x 1 gradients of input layer biases

end

