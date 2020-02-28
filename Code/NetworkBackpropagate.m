function WNew = NetworkBackpropagate(I, W, eta)
    % I is the cell array of neuron values
    % W is the cell array of adjacency matrices containing edge weights
    % WNew is the updated W, after backpropagation (adjusting weights to minize error)
    % Eta is the training rate
    wOutputOld = W{end}; % Used to calculate the other deltas
    
    % Calculate the weights of W{end}
    % Calculate error
    % Calculate delta using error and I
    % Calculate delta W using delta, I, eta
    % W{end} = delta * eta * I{}
    
    
    
    for i = length(W)-1:-1:1
        % Calculate all the weights for the other layers W{i}
        
    end
end