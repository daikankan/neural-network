clc; clear; close;

load train_VectorInputs;
load train_Targets;

load test_VectorInputs;
load test_Targets;

[net] = newnet([300], train_VectorInputs, train_Targets); % creat a new network

[net] = initnet(net); % initial weights

%% Training

traintime(1) = 0;
for j = 1:300
    
    tic;
    [net] = traingd(net, train_VectorInputs, train_Targets, 1, 0.01, 0.8); % gradient-based training
    time = toc;
    
    traintime(j+1) = traintime(j) + time; % training time of vector inputs
    
    train_Error = 0; % misclassification
    for i = 1 : size(train_Targets, 2)
        
        [x, label_IndexExpected] = max(train_Targets(:, i));
        
        node = forward(net, train_VectorInputs(:, i));
        
        [x, label_IndexActual] = max(node{net.nolayers});
        
        if label_IndexActual ~= label_IndexExpected
            train_Error = train_Error + 1;
        end
    end
    
    TrainAccuracyRateV(j) = 1 - train_Error/size(train_Targets, 2);
    
    %% Testing
    
    test_Error = 0; % misclassification
    for i = 1 : size(test_Targets, 2)
        
        [x, label_IndexExpected] = max(test_Targets(:, i));
        
        node = forward(net, test_VectorInputs(:, i));
        
        [x, label_IndexActual] = max(node{net.nolayers});
        
        if label_IndexActual ~= label_IndexExpected
            test_Error = test_Error + 1;
        end
    end
    
    TestAccuracyRateV(j) = 1 - test_Error/size(test_Targets, 2);
    
end

traintimeV = traintime(2:end);

save TrainAccuracyRateV TrainAccuracyRateV;
save TestAccuracyRateV TestAccuracyRateV;
save traintimeV traintimeV;

figure(1)
plot(TrainAccuracyRateV);
figure(2)
plot(TestAccuracyRateV);
figure(3)
plot(traintimeV);