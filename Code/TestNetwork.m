function [error, prediction] = TestNetwork(testData, testLabels, WFinal)
% Calculate values error and predictions
%   This functions calculates the error from the test data by comparing the
%   output values to the labels.
    nTests = length(testLabels);
    prediction = zeros(nTests,1);
    error = zeros(nTests,1);
    for k = 1:nTests
        I = cell(4,1);
        I{1} = testData(:,k);
        for i = 2:4
            [~, nNeurons] = size(WFinal{i-1});
            I{i} = zeros(nNeurons,1);
            for j = 1:nNeurons
                I{i}(j) = Neuron(I{i-1},WFinal{i-1}(:,j));
            end
        end
        prediction(k) = find(I{end}==max(I{end})) - 1;
        error(k) = 1 - I{end}(testLabels(k)+1);
    end
end

