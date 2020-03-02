function WNew = NetworkBackpropagateOldWeights(I, W, label, eta)
    % I is the cell array of neuron values
    % W is the cell array of adjacency matrices containing edge weights
    % WNew is the updated W, after backpropagation (adjusting weights to minize error)
    % Eta is the training rate
    delta = cell(1,length(W));
    error = cell(1,length(W));
    
    % First calculate errors and deltas
    target = zeros(length(I{end}),1);
    target(label+1) = 1; % set the index representing the actual value to 1
    error{end} = (I{end}(:) - target)'; % error as a vector
    delta{end} = (I{end}(:).*(ones(length(I{end}),1)-I{end}(:)).*error{end}(:))';
    
    % Then psuedo_errors and delta for hidden layers
    for L = length(W)-1:-1:1
        [~, q] = size(W{L});
        error{L} = delta{L+1}*W{L+1}'; % vectorized (delta{L+1} is at Layer L=L+2 because of way delta is defined)
        delta{L} = I{L+1}'.*(ones(1,q)-I{L+1}').*error{L};
    end
    
    % Calculate delta W using delta, I, eta
    for L = length(W):-1:1
        deltaW = eta*I{L}*delta{L}; % I is already px1 and delta is 1xq
        W{L} = W{L} + deltaW;
    end
    
    WNew = W;
end