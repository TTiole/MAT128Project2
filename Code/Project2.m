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
hidden_layers = 2;
layers = hidden_layers + 2;
neurons = [400 10 5 10]; % just as example with 400 inputs, 10 neurons is first hidden layer, 5 in second and 10 outputs

% Not sure if this is headed in the right direction
Layer = cell(layers,1);
Layer{1} = reshape(img_train(:,:,1),400,1);
for i = 2:layers
    for j = 1:neurons(i)
        Layer{i}(j) = Neuron( Layer{i-1}, W{i-1}(:,j) );
    end
end
