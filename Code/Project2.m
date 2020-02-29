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
plot_data(img_test(:,:,1))
xlim([0 20]); ylim([0 20]); axis off;
hold off

%% Initialize Values
neurons = [400 10 5 10]; % just as example with 400 inputs, 10 neurons is first hidden layer, 5 in second and 10 outputs
layers = length(neurons);
Network(neurons, img_train, img_test, label_train, .1, .05);
