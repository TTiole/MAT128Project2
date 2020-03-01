function WNew = NetworkBackpropagateOldWeights(I, W, label, eta)
    % I is the cell array of neuron values
    % W is the cell array of adjacency matrices containing edge weights
    % WNew is the updated W, after backpropagation (adjusting weights to minize error)
    % Eta is the training rate
    wOutputOld = W{end}; % Used to calculate the other deltas
    delta = cell(1,length(W));
    error = cell(1,length(W));
    
    % First calculate errors and deltas
    target = zeros(length(I{end}),1);
    target(label+1) = 1; % set the index representing the actual value to 1
    error{end} = I{end}(:) - target; % error as a vector
    delta{end} = I{end}(:).*(ones(length(I{end}),1)-I{end}(:)).*error{end}(:);
    
    % Then psuedo_errors and delta for hidden layers
    for L = length(W)-1:-1:1
        [~, q] = size(W{L});
        for i = 1:q
            error{L}(i) = sum(delta{L+1}(:)'.*W{L+1}(i,:));
            delta{L}(i) = I{L+1}(i)*(1-I{L+1}(i))*error{L}(i);
        end
    end
    
    % Calculate delta W using delta, I, eta
    for L = length(W):-1:1
        [p,q] = size(W{L});
        % Caclulate delta W using delta, I, eta (think about if this can be vectorized)
        deltaW = zeros(p,q);
        for i = 1:p
            for j = 1:q
                deltaW(i,j) = eta*delta{L}(j)*I{L}(i);
            end
        end
        W{L} = W{L} + deltaW;
    end

    WNew = W;
end