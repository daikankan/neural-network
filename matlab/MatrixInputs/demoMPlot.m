clc; clear; close;

load train_MatrixInputs;
load train_Targets;

load test_MatrixInputs;
load test_Targets;

[net] = newnet([300], train_MatrixInputs, train_Targets); % creat a new network

[net] = initnet(net); % initial weights

%% Training

traintime(1) = 0;
for j = 1:300
    
    tic;
    [net] = traingd(net, train_MatrixInputs, train_Targets, 1, 0.01, 0.8); % gradient-based training
    time = toc;
    
    traintime(j+1) = traintime(j) + time; % training time of matrix inputs
    
    train_Error = 0; % misclassification
    for i = 1 : size(train_Targets, 2)
        
        [x, label_IndexExpected] = max(train_Targets(:, i));
        
        node = forward(net, train_MatrixInputs(:, :, i));
        
        [x, label_IndexActual] = max(node{net.nolayers});
        
        if label_IndexActual ~= label_IndexExpected
            train_Error = train_Error + 1;
        end
    end
    
    TrainAccuracyRateM(j) = 1 - train_Error/size(train_Targets, 2);
    
    %% Testing
    
    test_Error = 0; % misclassification
    for i = 1 : size(test_Targets, 2)
        
        [x, label_IndexExpected] = max(test_Targets(:, i));
        
        node = forward(net, test_MatrixInputs(:, :, i));
        
        [x, label_IndexActual] = max(node{net.nolayers});
        
        if label_IndexActual ~= label_IndexExpected
            test_Error = test_Error + 1;
        end
    end
    
    TestAccuracyRateM(j) = 1 - test_Error/size(test_Targets, 2);
    
end

traintimeM = traintime(2:end);

save TrainAccuracyRateM TrainAccuracyRateM;
save TestAccuracyRateM TestAccuracyRateM;
save traintimeM traintimeM;

figure(1)
plot(TrainAccuracyRateM);
figure(2)
plot(TestAccuracyRateM);
figure(3)
plot(traintimeM);