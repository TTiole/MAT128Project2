function [err, prediction, WFinal] = Network(nNeurons, trainingData, testData, trainLabels, testLabels, weightScale, eta)
    % This function goes through the entire process of training and
    % testing a network
    
    % err is the total error during the testing phase
    % WFinal is the set of weights at the end
    % nNeurons is an array of the number of neurons the network should have per layer
    % (number of layers is implied by the length of the array)
    % weightScale is the range of the random initial weights
    % eta is the 
    
    nLayers = length(nNeurons);
    
    % Preallocate
    wInitial = cell(1, nLayers-1);
    
    % Initialize weights
    for i = 1:nLayers-1
        wInitial{i} = -1 + 2*weightScale*rand(nNeurons(i), nNeurons(i+1));
    end
    
    % Train data
    W = wInitial;
    trainingDataLength = size(trainingData,2);
    for i = 1:trainingDataLength
        [~, W] = NetworkIteration(W, trainingData(:, i), trainLabels(i), eta, nNeurons);
    end
    WFinal = W;
    
    % Test data
    [err, prediction] = TestNetwork(testData, testLabels, WFinal, nNeurons);
end