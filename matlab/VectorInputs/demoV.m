clc; clear; close;

%% Training

load train_VectorInputs;
load train_Targets;

[net] = newnet([300], train_VectorInputs, train_Targets); % creat a new network

[net] = initnet(net); % initial weights

tic
[net] = traingd(net, train_VectorInputs, train_Targets, 100, 0.01, 0.8); % gradient-based training
toc

traintimeM = toc; % training time of matrix inputs

train_Error = 0; % misclassification
for i = 1 : size(train_Targets, 2)
    
    [x, label_IndexExpected] = max(train_Targets(:, i));
    
    node = forward(net, train_VectorInputs(:, i));
    
    [x, label_IndexActual] = max(node{net.nolayers});
    
    if label_IndexActual ~= label_IndexExpected
        train_Error = train_Error + 1;
    end
end

TrainAccuracyRate = 1 - train_Error/size(train_Targets, 2)

%% Testing

load test_VectorInputs;
load test_Targets;

test_Error = 0; % misclassification
for i = 1 : size(test_Targets, 2)
    
    [x, label_IndexExpected] = max(test_Targets(:, i));
    
    node = forward(net, test_VectorInputs(:, i));
    
    [x, label_IndexActual] = max(node{net.nolayers});
    
    if label_IndexActual ~= label_IndexExpected
        test_Error = test_Error + 1;
    end
end

TestAccuracyRate = 1 - test_Error/size(test_Targets, 2)