function [ node ] = forward( net, p )
%FORWARD Summary of this function goes here
%   Detailed explanation goes here

node = {}; % each layer's nodes vector

uxv = sum( net.weights{1, 1}(:, 1:net.inputm) * p .* net.weights{1, 1}(:, net.inputm + 1:end), 2); 
uxvb = uxv + net.weights{1, 2};
node{1} = 1 ./ (1 + exp( - uxvb)); % h x 1 input layer nodes

wx = net.weights{2, 1} * node{1};
wxb = wx + net.weights{2, 2};
node{2} = wxb; % o x 1 output layer nodes
end

