function [error, avgError, prediction, correctness, avgCorrectness] = TestNetwork(testData, testLabels, WFinal, nNeurons)
% Goes through the test data and verifies whether the neural network is
% working

    % @ INPUT
    % testData is the flattened test images
    % testLabels are the labels corresponding to the images of testData
    % WFinal are the final weights (they will no longer change)
    % nNeurons is the array containing the number of neurons per layer
    
    % @ OUTPUT
    % error is the array containing 1-confidence of the network for each image
    % avgError is the average of the aforementioned array
    % prediction is the array containing the prediction of the network per image
    % correctness is a logical array indicating which predictions the network got right
    % avgCorrectness is the percentage of the time the network was right
    
    % Number of tests to run
    nTests = length(testLabels);
    
    % Preallocate
    prediction = zeros(nTests,1);
    error = zeros(nTests,1);
    
    % Calculate the output for each test
    for k = 1:nTests
        I = ForwardPass(nNeurons, testData(:,k), WFinal);
        
        % Take the neuron that fired the most in the output layer as the
        % predicted value
        prediction(k) = find(I{end}==max(I{end})) - 1;
        
        % Validate against the label
        error(k) = 1 - I{end}(testLabels(k)+1);
    end

    % Indicate which predictions were correct
    correctness = prediction == testLabels;
    
    % Calculate the average values
    avgError = mean(error);
    avgCorrectness = mean(correctness);
end

