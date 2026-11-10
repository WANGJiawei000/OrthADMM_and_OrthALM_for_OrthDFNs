function [U0, V0, output] = NN_fp(x_train, W0, b0, act_type, leak_slope)
%NN_FP Forward propagation for the OrthDNN model.

if nargin < 5 || isempty(leak_slope)
    leak_slope = 0.05;
end

L = numel(W0) - 1;
U0(L+1).U = [];
V0(L+1).V = [];

U0(1).U = W0(1).W * x_train + b0(1).b;
V0(1).V = act_fun(U0(1).U, act_type, leak_slope);

for i = 2:L
    U0(i).U = W0(i).W * V0(i-1).V + b0(i).b;
    V0(i).V = act_fun(U0(i).U, act_type, leak_slope);
end

U0(L+1).U = W0(L+1).W * V0(L).V + b0(L+1).b;
V0(L+1).V = act_fun(U0(L+1).U, act_type, leak_slope);

output = V0(L+1).V;
end
