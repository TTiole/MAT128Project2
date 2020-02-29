function WNew = NetworkBackpropagate(I, W, label, eta)
    % I is the cell array of neuron values
    % W is the cell array of adjacency matrices containing edge weights
    % WNew is the updated W, after backpropagation (adjusting weights to minize error)
    % Eta is the training rate
    wOutputOld = W{end}; % Used to calculate the other deltas
    
    % Calculate the weights of W{end} (between the last hidden layer and the output
    % Calculate error
    target = zeros(10,1); % 10 for the 10 digits ( could make this length(output) if we want to generalize that far)
    target(label+1) = 1; % set the index representing the actual value to 1
    error = I{end}(:) - target; % error as a vector
    % Calculate delta using error and I
    deltaEnd = I{end}(:).*(ones(length(I{end}))-I{end}(:)).*error(:); % colons arent necessary just emphasizing the form for now
    % Calculate delta W using delta, I, eta
    [p, q] = size(W{end});
    deltaW = zeros(p,q);
    for i = 1:p
        for j = 1:q
            deltaW(i,j) = eta*deltaEnd(j)*I{end-1}(i);
        end
    end
    W{end} = W{end} + deltaW;  % W{end} = delta * eta * I{}
    
    for L = length(W)-1:-1:1
        [p, q] = size(W{L}); % p is the layer being update and q is the previous layer ( i think )
        % Calculate all the weights for the other layers W{i}
        % Calculate delta
        % for i = 1:p
          %  delta(i) = I{L-1}(i)*(1-I{L-1}(i))*sum(deltaEnd*
        % delta = deltaEnd*wOutputOld'*I{L}.*( ones(length(I{L})) - I{L} ); % I believe this is what it should be, just use hte output layer wieghts
        delta = dot(deltaEnd,wOutputOld)*I{L}.*( ones(length(I{L})) - I{L} );
        % Caclulate delta W using delta, I, eta (think about if this can be vectorized)
        deltaW = zeros(p,q);
        for i = 1:p
            for j = 1:q
                deltaW(i,j) = eta*delta(j)*I{L-1}(i);
            end
        end
        W{L} = W{L} + deltaW;
    end
end