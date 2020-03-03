function [result, WNew] = NetworkIteration(W, input, label, eta, nNeurons)
    % This function runs through a single iteration of the network (a
    % single image)
    
    % W is the cell array of adjacency matrices containing the edge weights
    % input is the image as a 2D array of values between 0 and 1
    % Caller of this function is expected to keep track of the error rate
    
    I = ForwardPass(nNeurons,input,W);
    result = I{end};
    
    % Backpropagation
    WNew = NetworkBackpropagate(I, W, label, eta);
end
