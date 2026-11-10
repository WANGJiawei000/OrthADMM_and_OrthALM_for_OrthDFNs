function f = act_fun(u, act_type, leak_slope)
%ACT_FUN Activation function used by the OrthDNN ADMM code.
% act_type: 1 sigmoid, 2 ReLU, 3 leaky ReLU, 4 linear, 5 sign.

if nargin < 3 || isempty(leak_slope)
    leak_slope = 0.05;
end

switch act_type
    case 1
        f = 1 ./ (1 + exp(-u));
    case 2
        f = max(u, 0);
    case 3
        f = max(u, 0) + leak_slope * min(u, 0);
    case 4
        f = u;
    case 5
        f = sign(u);
    otherwise
        error('Unknown act_type. Use 1 sigmoid, 2 ReLU, 3 leaky ReLU, 4 linear, or 5 sign.');
end
end
