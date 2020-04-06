%% PerceptronWeights.m
% Program to take in a set of data and return to optimal, least squares
% weight solution.
%
% INPUTS
% X - data P patterns x N neurons
% Y - P patterns either 1 or 0
% Reg - the exponent regularisor to use, i.e. L2 put reg = 2
% Lambda - the parameter on the regularisation
%
% OUTPUTS
% W - the weight vector N long
%
function[W] = PerceptronWeights(X, Y, Reg, Lambda)

Datapoints = length(Y);
Neurons = size(X, 2);
Y(Y == 0) = -1;
if Reg == 0
    if Datapoints < Neurons
        Q = 1/Neurons*(X*X');
        W = 1/Neurons*Y'*(Q\X);
        W = W';
    else
        W = (X'*X)\X'*Y;
    end
elseif Reg == 2
    W = (X'*X + Lambda*eye(Neurons))\X'*Y;
elseif Reg == 1
    fun = @(W)-W*(Y'*X)'*(1/Datapoints) + 1/(2*Datapoints)*W*(X'*X)*W' + Lambda*norm(W,1);
    W0 = randn([1,Neurons]);
    [W,fval] = fminunc(fun,W0);
end