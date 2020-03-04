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

%% Plot numbers
% plot_data  = @(A) image(rot90(A')*100);
% 
% figure(); hold on
% plot_data(img_test(:,:,12))
% xlim([0 20]); ylim([0 20]); axis off;
% hold off

%% Testing
clc
neurons = [400 13 7 10]; % Define neurons and layers
tic
[err, avgError, prediction, WFinal, correctness, avgCorrectness] = Network(neurons, trainVector(:,1:n_train), testVector, label_train(1:n_train), label_test, .1, .5);

fprintf("The network executed with an average error of %2.2f%% and average correctness of %2.2f%% \n", avgError*100, avgCorrectness*100);

%% Parameter study
neuronPar = {[400  13 7  10];
             [400 10 10 10];
             [400 4  8   5  10]};
          
trainSets = [n_train/2 n_train/2 n_train /10 ]; %n_train/60];
weightScales = linspace(.01,1.5,10);
trainRates = linspace(0.01,1.5,10);

iter = 0;
tic
avgError = zeros(length(neuronPar),length(weightScales),length(trainRates));
avgCorrectness = avgError;
for i = 1:1 %length(trainSets)
    for j = 1:length(neuronPar)
        for k = 1:length(weightScales)
            for l = 1:length(trainRates)
                iter = iter + 1;
                fprintf('On iteration %i of %i ',iter,length(neuronPar)*length(weightScales)*length(trainRates))
                tic
                [~, avgError(j,k,l), ~, ~, ~, avgCorrectness(j,k,l)] = Network(neuronPar{j}, trainVector(:,1:trainSets(i)), testVector, label_train(1:trainSets(i)), label_test, weightScales(k), trainRates(l));
                toc
            end
        end
    end
end
toc


%% Plotting the results of parameter study
% The networds are connected by a meshgrid of the same color
colors = {'b', 'g', 'r', 'c', 'm', 'y', 'b'};
[X,Y] = meshgrid(trainRates,weightScales);
figure(); hold on
for i = 1:length(neuronPar)
    Z = avgCorrectness(i,:,:);
    size(Z)
    Z = reshape(Z,[length(weightScales),length(trainRates)]);
    size(Z)
    s = meshz(X,Y,Z);
    s.EdgeColor = colors{i};
    lstring{i} = strcat('Network ',num2str(i));
end
zlabel('Correctness')
ylabel('Weights')
xlabel('Train Rates')
legend(lstring)
hold off

% Once a network is picked then we can refine weights and rates
% once rates and weights are refined, we can look at variation in training
% images and then possibly bias
%% Old Plot Ideas
% [X,Y] = meshgrid(-3:.125:3);
% Z = peaks(X,Y);
% C = gradient(Z);
% meshz(X,Y,Z,C)
% colorbar
% clc
% [X,Y,Z] = meshgrid(1:3,weightScales,trainRates(1:3));
% C = avgCorrectness(:,:,1:3);
% figure(); hold on
% colorbar
% meshz(X,Y,Z,C);
% xlabel('Networks')
% ylabel('Weight Scales')
% zlabel('Train Rates')
% hold off

% %%
% x = 0:.1:1;  y = x; z = x;
% [X,Y,Z] = meshgrid(x,y,z);
% mask = X.^2 + Y.^2 + Z.^2 <= 1;
% a = X(mask);
% F = sqrt(1-(X.^2+Y.^2+Z.^2));   %caution, need this order because of round-off
% figure(); hold on
% scatter3(X(mask),Y(mask),Z(mask),20,F(mask))
% colormap
% hold off
% subplot(1,2,2);
% F2 = F;
% F2(~mask) = nan;
% for level = 0.2:0.2:0.8
%     isosurface(X, Y, Z, F2, level);
% end
%figure(); hold on;
%plot(nTrains,error) % plot how the error changes with the number of
%traninng used
%hold off

%%
% figure(); hold on;
% contour3(avgCorrectness(:,:,3))
% xlabel('Networks')
% ylabel('Weights')
% zlabel('Correctness')
% colorbar
% hold off;
% 
% [X,Y,Z] = meshgrid(1:3,weightScales,trainRates);
% mask = 1:length(neuronPar)*length(weightScales)*length(trainRates);
% figure(); hold on
% colorbar
% colormap(jet)
% scatter3(X(mask),Y(mask),Z(mask),20,avgCorrectness(mask));
% xlabel('Networks')
% ylabel('Weight Scales')
% zlabel('Train Rates')
% for i = mask
%     text(X(i),Y(i),Z(i),num2str(avgCorrectness(i)))
% end
% hold off