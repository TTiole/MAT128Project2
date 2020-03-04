function [I] = ForwardPass(nNeurons, input, W)
%ForwardPass takes in the input and weights to calculate the values at each
%neuron

    % @ INPUT
%   nNeurons is a vector of the number of neurons for each layer
%   input is the values of the input layer
%   W is a cell array of the weights (represented as adjacency matrices)
    
    % @ OUTPUT
    % I is the cell array containing a vector of neuron values per layer

    nLayers = length(nNeurons);
    I = cell(nLayers,1);
    I{1} = input;
    for i = 2:nLayers
        I{i} = zeros(nNeurons(i),1);
        for j = 1:nNeurons(i)
            I{i}(j) = Neuron(I{i-1},W{i-1}(:,j));
        end
    end

end

