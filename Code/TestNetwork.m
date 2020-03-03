function [error, prediction] = TestNetwork(testData, testLabels, WFinal, nNeurons)
% Calculate values error and predictions
%   This functions calculates the error from the test data by comparing the
%   output values to the labels.

    nTests = length(testLabels);
    prediction = zeros(nTests,1);
    error = zeros(nTests,1);
    for k = 1:nTests
        I = ForwardPass(nNeurons, testData(:,k), WFinal);
        prediction(k) = find(I{end}==max(I{end})) - 1;
        error(k) = 1 - I{end}(testLabels(k)+1);
    end
    
end

