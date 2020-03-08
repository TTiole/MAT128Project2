% MAT128B Project 2
% Winter 2020
% Created by: Eli Frances Nikos
clc; close all;

%% Read Data
n_train = 60000; n_test = 10000;
[img_train, label_train] = readMNIST('../Data/train-images.idx3-ubyte','../Data/train-labels.idx1-ubyte',n_train,0);
[img_test , label_test]  = readMNIST('../Data/t10k-images.idx3-ubyte' ,'../Data/t10k-labels.idx1-ubyte' ,n_test,0);
% flatten data
trainVector = reshape(img_train,[400,n_train]);
testVector = reshape(img_test,[400,n_test]);

%% Parameter study
neuronPar = {[400 13 7  10];
             [400 10 10 10];
             [400 5  10];
             [400 15 10];
             [400 20 10];
             [400 40 10];
             [400 100 25 5 10];
             [400 100 10];
             [400 200 10];
             [400 15 15 10];
             [400 20 20 10]
             [400 40 20 10];
             [400 40 30 20 10];
             [400 20 20 10 10];
             [400 40 20 10 10]};
             %[400 4  8  5  10];

trainSets = [n_train n_train/2 n_train/10 n_train/60];
weightScales = linspace(0.01,1.5,10);
trainRates = linspace(0.01,1.5,10);
clear trainSets
trainSets = n_train/5; % For faster parameter study
% looked at some convergences and there seems to be little gain after this amount
% will use the full amount once a configuration is chosen

iter = 0;
avgError = zeros(length(neuronPar),length(weightScales),length(trainRates));
avgCorrectness = avgError;
for i = 1:length(trainSets)
    for j = 1:length(neuronPar)
        for k = 1:length(weightScales)
            for l = 1:length(trainRates)
                iter = iter + 1;
                fprintf('On iteration %i of %i ',iter,length(neuronPar)*length(weightScales)*length(trainRates)*length(trainSets))
                tic
                [~, avgError(j,k,l), ~, ~, ~, avgCorrectness(j,k,l)] = Network(neuronPar{j}, trainVector(:,1:trainSets(i)), testVector, label_train(1:trainSets(i)), label_test, weightScales(k), trainRates(l));
                toc
            end
        end
    end
end

%%
%save('Param6.mat','avgCorrectness','avgError','neuronPar','weightScales','trainRates')


%% Plotting the results of parameter study
colors = [1 1 0;
          1 0 1;
          0 1 1;
          1 0 0;
          0 1 0;
          0 0 1;
          0 0 0;
          0.25 0.25 0.25;
          0 0.5 0;
          0 0.75 0.75;
          0.75 0 0.75;
          0.75 0.75 0; 
          .5 .5 .5;
          .75 .25 .25;
          .5 0 0];
[X,Y] = meshgrid(trainRates,weightScales);
figure(); hold on
for i = 1:length(neuronPar)
    Z = avgCorrectness(i,:,:);
    size(Z)
    Z = reshape(Z,[length(weightScales),length(trainRates)]);
    size(Z)
    s = meshz(X,Y,Z);
    s.EdgeColor = colors(i,:);
    lstring{i} = strcat('Network ',num2str(i));
end
zlabel('Correctness')
ylabel('Weights')
xlabel('Train Rates')
legend(lstring)
hold off
