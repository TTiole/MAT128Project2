% MAT128B Project 2
% Winter 2020
% Created by: Eli Frances Nikos
clc; close all;

%% Read Data
n_train = 60000; n_test = 10000;
[img_train, label_train] = readMNIST('../Data/train-images.idx3-ubyte','../Data/train-labels.idx1-ubyte',n_train,0);
[img_test , label_test]  = readMNIST('../Data/t10k-images.idx3-ubyte' ,'../Data/t10k-labels.idx1-ubyte' ,n_test,0);

%% Plot numbers
plot_data  = @(A) image(rot90(A')*100);

figure(); hold on
plot_data(img_test(:,:,12))
xlim([0 20]); ylim([0 20]); axis off;
hold off

%% Testing
clear I W WFinal
clc
neurons = [400 13 7 10]; % Define neurons and layers
% flatten data
trainVector = reshape(img_train,[400,n_train]);
testVector = reshape(img_test,[400,n_test]);
% n = 1000; % run a subset of data for debugging so it doesnt take as long
tic
[err, avgError, prediction, WFinal, correctness, avgCorrectness] = Network(neurons, trainVector(:,1:n_train), testVector(:,1:n_test), label_train(1:n_train), label_test(1:n_test), 1, .05);
toc

fprintf("The network executed with an average error of %2.2f%% and average correctness of %2.2f%% \n", avgError*100, avgCorrectness*100);

%% Parameter study
%figure(); hold on;
%plot(nTrains,error) % plot how the error changes with the number of
%traninng used
%hold off