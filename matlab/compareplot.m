%clear; clc;

load TrainAccuracyRateM;
load TestAccuracyRateM;
load traintimeM;

% load TrainAccuracyRateV;
% load TestAccuracyRateV;
% load traintimeV;

iter = 1:300;

figure(2)
plot(iter, traintimeM(1,iter)./3600, 'b');
hold on
plot(iter, traintimeV(1,iter)./3600, '--k');
legend('Matrix Inputs', 'Vector Inputs', 2);
set(get(gca,'YLabel'), 'String', 'Training Time (h)')
set(get(gca,'XLabel'), 'String', 'Iteration (With 300 Hidden Nodes)')

figure(3)
plot(iter, TrainAccuracyRateM(1,iter), 'b');
%hold on
% plot(iter, TrainAccuracyRateM1, 'b');
hold on
plot(iter, TrainAccuracyRateV(1,iter), '--k');
%hold on
%plot(iter, TrainAccuracyRateV1, '--k');
ylim([0.8,1])
legend('Matrix Inputs','Vector Inputs', 4);
set(get(gca,'YLabel'), 'String', 'Training Accuracy Rate')
set(get(gca,'XLabel'), 'String', 'Iteration (With Learning Rate 0.01 and Momentum 0.8)')

figure(4)
plot(iter, TestAccuracyRateM(1,iter), 'b');
%hold on
%plot(iter, TestAccuracyRateM1, 'b');
hold on
plot(iter, TestAccuracyRateV(1,iter), '--k');
%hold on
%plot(iter, TestAccuracyRateV1, '--k');
ylim([0.8,1])
legend('Matrix Inputs','Vector Inputs', 4);
set(get(gca,'YLabel'), 'String', 'Testing Accuracy Rate')
set(get(gca,'XLabel'), 'String', 'Iteration (With Learning Rate 0.01 Momentum 0.8)')