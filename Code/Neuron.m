function OUT = Neuron(I, W)
    NET = 0;
    n = length(I);
    for i = 1:n
        IW = I(i)*W(i);
        NET = NET + IW;
    end
    OUT = 1/(1 + exp(-NET));
end