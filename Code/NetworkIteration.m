function [result, WNew] = NetworkIteration(W, input, label, eta, nNeurons)
    % This function runs through a single iteration of the network (a
    % single image)
    
    % @ OUTPUT
    % result is the values of the neurons of the output layer
    % WNew is the new weights
    
    % @ INPUT
    % W is the cell array of adjacency matrices containing the edge weights
    % input is the image as a vector of values between 0 and 1
    % label is the expected number
    % eta is the training rate
    % nNeurons is the array with the number of neurons per layer
    
    % Do the forward pass to calculate all the neuron values
    I = ForwardPass(nNeurons,input,W);
    result = I{end};
    
    % Do the backpropagation to update the weights in order to minize error
    WNew = NetworkBackpropagate(I, W, label, eta);
end
