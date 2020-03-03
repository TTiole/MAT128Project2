function WNew = NetworkBackpropagate(I, W, label, eta)
    % This function goes through the backpropagation step of using the
    % label and the input values in order to change the weights in order to
    % minize error
    
    % @ INPUT
    % I is the cell array of neuron values
    % W is the cell array of adjacency matrices containing edge weights
    % label is the expected number
    % eta is the training rate
    
    % OUTPUT
    % WNew is the updated W, after backpropagation (adjusting weights to minize error)

    % Preallocate
    delta = cell(1,length(W));
    error = cell(1,length(W));
    
    % First calculate errors and deltas of the output layer
    target = zeros(length(I{end}),1);
    target(label+1) = 1; % set the index representing the actual value to 1
    error{end} = (I{end}(:) - target)'; % error as a vector
    delta{end} = (I{end}(:).*(ones(length(I{end}),1)-I{end}(:)).*error{end}(:))';
    
    % Then psuedo_errors and delta for hidden layers
    % Pseudo_errors represents the sum of deltas times weights of the
    % previous layer
    for L = length(W)-1:-1:1
        [~, q] = size(W{L});
        error{L} = delta{L+1}*W{L+1}'; % vectorized (delta{L+1} is at Layer L=L+2 because of way delta is defined)
        delta{L} = I{L+1}'.*(ones(1,q)-I{L+1}').*error{L};
    end
    
    % Calculate the change of W using the delta values of previous layer,
    % neuron values of current layer and eta
    for L = length(W):-1:1
        deltaW = eta*I{L}*delta{L}; % I is already px1 and delta is 1xq
        W{L} = W{L} + deltaW;
    end
    
    % Update the weights
    WNew = W;
end
