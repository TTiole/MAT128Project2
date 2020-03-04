function [err, avgError, prediction, WFinal, correctness, avgCorrectness] = Network(nNeurons, trainingData, testData, trainLabels, testLabels, weightScale, eta)
    % This function goes through the entire process of training and
    % testing a network
    
    % @ OUTPUT
    % err is the total error during the testing phase
    % avgError is the average of the total error
    % WFinal is the set of weights at the end
    % correctness is a logical array which indicates which numbers the network predicted correctly
    % avgCorrectness is the percentage of numbers the network predicted correctly
    
    % @ INPUT
    % nNeurons is an array of the number of neurons the network should have per layer
    % (number of layers is implied by the length of the array)
    % weightScale is the range of the random initial weights
    % eta is the training rate
    
    nLayers = length(nNeurons);
    
    % Preallocate
    wInitial = cell(1, nLayers-1);
    
    % Initialize weights
    for i = 1:nLayers-1
        wInitial{i} = -weightScale + 2*weightScale*rand(nNeurons(i), nNeurons(i+1));
    end
    W = wInitial;
    
    % W represents a cell array of adjacency matrices
    
    % Train data
    trainingDataLength = size(trainingData,2);
    for i = 1:trainingDataLength
        % Go through each training image
        [~, W] = NetworkIteration(W, trainingData(:, i), trainLabels(i), eta, nNeurons);
    end
    WFinal = W;
    
    % Test data
    [err, avgError, prediction, correctness, avgCorrectness] = TestNetwork(testData, testLabels, WFinal, nNeurons);
end