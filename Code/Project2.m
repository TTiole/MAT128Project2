% MAT128B Project 2
% Winter 2020
% Created by: Eli Frances Nikos


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
neurons = [400 10 5 1]; % just as example with 400 inputs, 10 neurons is first hidden layer, 5 in second and 1 output
W = cell(layers,1);
scale = .5; offset = 0; % these may be unnecessary
for i = 1:layers-1
    W{i} = offset + scale*rand(neurons(i),neurons(i+1)); % rand randomly generates values from 0 to 1, so use scale to effect value range
end