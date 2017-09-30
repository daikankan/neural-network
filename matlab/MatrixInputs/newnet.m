function [ net ] = newnet( H, P, T )
%newnet is used to create a feedforward neural network.
%   net is a feedforward neural network with all its weights are set to zero.
%   H is the number of nodes of each layer except the output layer. 
%   P and T are the input and targets training vectors, respectively.
%
% Dai Kankan 2014.

net.nolayers = size(H, 2) + 1; % number of layers
net.nonodes = [H, size(T, 1)]; % number of nodes of each layer

net.inputm = size(P(:, :, 1), 1); % the number of raws of input matrix
net.inputn = size(P(:, :, 1), 2); % the number of colums s of input matrix

net.weights = {};

net.weights{1, 1} = zeros(H(1), net.inputm + net.inputn); % input layer weights

net.weights{1, 2} = zeros(H(1), 1); % input layer biases

net.weights{2, 1} = zeros(size(T, 1), H(1)); % output layer weights

net.weights{2, 2} = zeros(size(T, 1), 1); % output layer biases

disp('New network has been created.')
end
