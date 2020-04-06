% A function to threshold to a particular sparseness a representation

function [X, phi] = Threshold(X, phi, S, Thresh_Option, alpha)
% Thresh_Option chooses between enforce sparseness over whole population,
% or over individual neurons. 0 is over pop, 1 is over neurons.
% 2 is now applying a sigmoid that ensures only sparseness proportion of
% the data are above 0.5, in simple circumstances. 
%   In this situation alpha sets constant on sigmoid

if ~Thresh_Option
    % Case one: threshold given and no sparseness, just apply this threshold
    if exist('phi', 'var') && ~exist('S','var')
        X(abs(X)<phi) = 0;
        X(abs(X)>0) = X(abs(X)>0) - sign(X(abs(X)>0)).*phi;
    % Case two: sparseness level set but no threshold, apply sparseness
    elseif exist('S','var') && phi == 0
        temp = sort(reshape(abs(X),numel(X),1),'descend');
        loc = round(numel(X)*S);
        phi = temp(loc);
        X(abs(X)<phi) = 0;
        X(abs(X)>0) = X(abs(X)>0) - sign(X(abs(X)>0)).*phi;
    % Case three: Sparsness level set and a threshold, must rescale rep!
    else
        temp = sort(reshape(abs(X),numel(X),1),'descend');
        loc = round(numel(X)*S);
        Xi = phi/temp(loc);
        X = X.*Xi;
        X(abs(X)<phi) = 0;
        X(abs(X)>0) = X(abs(X)>0) - sign(X(abs(X)>0)).*phi;
    end
elseif Thresh_Option == 1
    Neurons = size(X, 1);
    for Neuron = 1:Neurons
        Activities = X(Neuron, :);
        temp = sort(Activities, 'descend');
        loc = max(1,round(Neurons*S));
        phi = temp(loc);
        Activities(Activities < phi) = 0;
        Activities(Activities > phi) = Activities(Activities > phi) - phi*ones(size(Activities(Activities>phi)));
        X(Neuron, :) = Activities;
    end
elseif Thresh_Option == 2
    % Assumes weight are normal with std dev 1 and input dim is 2
    Dim = 2;
    phi = sqrt(Dim/12)*erfcinv(2*S);
    X = 1./(1+ exp(-alpha.*(X - repmat(phi, size(X)))));
end