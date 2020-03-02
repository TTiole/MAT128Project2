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
plot_data(img_test(:,:,5))
xlim([0 20]); ylim([0 20]); axis off;
hold off

%% Testing
clear I W WFinal
clc
neurons = [400 10 10 10]; % just as example with 400 inputs, 10 neurons is first hidden layer, 5 in second and 10 outputs
layers = length(neurons);
train_New = reshape(img_train,[400,n_train]);
test_New = reshape(img_test,[400,n_test]);
n = 1000; % run a subset of data for debugging so it doesnt take as long
tic
[err, prediction, WFinal] = Network(neurons, train_New(:,1:n), img_test(:,1:n), label_train(1:n), label_test(1:n), .5, .05);
toc
