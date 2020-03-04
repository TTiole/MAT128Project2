% MAT128B Project 2
% Winter 2020
% Created by: Eli Frances Nikos
clc; close all;

%% Read Data
n_train = 60000; n_test = 10000;
[img_train, label_train] = readMNIST('../Data/train-images.idx3-ubyte','../Data/train-labels.idx1-ubyte',n_train,0);
[img_test , label_test]  = readMNIST('../Data/t10k-images.idx3-ubyte' ,'../Data/t10k-labels.idx1-ubyte' ,n_test,0);

%% Plot numbers
% plot_data  = @(A) image(rot90(A')*100);
% 
% figure(); hold on
% plot_data(img_test(:,:,12))
% xlim([0 20]); ylim([0 20]); axis off;
% hold off

%% Testing
clc
neurons = [400 8 13 10]; % Define neurons and layers
% flatten data
trainVector = reshape(img_train,[400,n_train]);
testVector = reshape(img_test,[400,n_test]);
tic
[err, avgError, prediction, WFinal, correctness, avgCorrectness] = Network(neurons, trainVector(:,1:n_train), testVector(:,1:n_test), label_train(1:n_train), label_test(1:n_test), .2, .05);
toc

fprintf("The network executed with an average error of %2.2f%% and average correctness of %2.2f%% \n", avgError*100, avgCorrectness*100);

% %% Parameter study
% neuronPar = {[400 13 7  10];
%               [400 10 10 10];
%               [400 3  5   3  10];
%               [400 20 20 10]};
%           
% trainSets = [n_train n_train/2 n_train /10 n_train/60];
% weightScales = linspace(.5,1.5,3);
% trainRates = linspace(0.01,1,4);
% 
% iter = 0;
% tic
% for i = 1:length(neuronPar)
%     for j = 1:length(trainSets)
%         for k = 1:length(weightScales)
%             for l = 1:length(trainRates)
%                 iter = iter + 1;
%                 fprintf('On iteration %i of %i',iter,length(neuronPar)*length(trainSets)*length(weightScales)*length(trainRates))
%                 tic
%                 Network(neuronPar{i}, trainVector(:,1:trainSets(j)), testVector, label_train(1:trainSets(j)), label_test, weightScales(k), trainRates(l));
%                 toc
%             end
%         end
%     end
% end
% toc

%figure(); hold on;
%plot(nTrains,error) % plot how the error changes with the number of
%traninng used
%hold off