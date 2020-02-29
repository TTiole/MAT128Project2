function [result, WNew] = NetworkIteration(W, input, training)
    % This function runs through a single iteration of the network (a
    % single image)
    
    % W is the cell array of adjacency matrices containing the edge weights
    % input is the image as a 2D array of values between 0 and 1
    % training is a boolean. If true, the network will go through a
    % backpropagation step and WNew will contain the new weights. If false
    % then that step will be skipped and WNew will be null. Caller of this
    % function is expected to keep track of the error rate
    
    nLayers = length(W)+1;
    I = cell(1, nLayers); % Values of each neuron
    % Initialize each layer with the right number of neurons
    for i = 1:nLayers-1
        I{i+1} = zeros(size(W{i},2), 1);
    end
    I{1} = input(:);
    
    % Begin populating the I array
    for i = 2:nLayers
        nNeurons = I{i};
        w = W{i-1};
        for j = 1:nNeurons
            I{i}(j) = Neuron(I{i-1}, w(:, j));
        end
    end
    
    result = I{end};
    if(~traning)
        WNew = NaN;
        return;
    end
    
    % Backpropagation
end
