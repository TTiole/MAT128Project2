function [I] = ForwardPass(nNeurons, input, W)
%ForwardPass takes in the input and weights to calculate the values at each
%neuron
%   nNeurons is a vector of the number of neurons for each layer
%   input is the values of the input layer
%   W is a cell array of the weights

    nLayers = length(nNeurons);
    I = cell(nLayers,1);
    I{1} = input;
    for i = 2:4
        I{i} = zeros(nNeurons(i),1);
        for j = 1:nNeurons(i)
            I{i}(j) = Neuron(I{i-1},W{i-1}(:,j));
        end
    end

end

