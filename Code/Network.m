function [err, prediction, WFinal] = Network(nNeurons, trainingData, testData, label, testLabels, weightScale, eta)
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
        % Initialize initial weights here. Ideally calling another function for
        % it (part v)
        wInitial{i} = zeros(nNeurons(i), nNeurons(i+1)); % not sure if this is necessary, i think rand might preallocate on its own
        wInitial{i} = weightScale*rand(nNeurons(i), nNeurons(i+1));
    end
    
    % Train data
    W = wInitial;
    %trainingDataLength = size(trainingData, 3);
    trainingDataLength = size(trainingData,2); % if we pre-flatten data into a vector
    for i = 1:trainingDataLength
        %[result, W] = NetworkIteration(W, trainingData(:, :, i), label, eta, true);
        [result, W] = NetworkIteration(W, trainingData(:, i), label(i), eta, true); % for preflatenned data
    end
    WFinal = W;
    [err, prediction] = TestNetwork(testData, testLabels, WFinal);
    %err = 0; % just so i can test values for now
end