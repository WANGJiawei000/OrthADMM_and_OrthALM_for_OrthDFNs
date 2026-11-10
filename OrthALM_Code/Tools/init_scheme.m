function [W0,b0] = init_scheme(dim, init_type, scal_factor)
%INIT_SCHEME Initialization schemes for the OrthDNN code.
% init_type: 0 zero, 1 Orth-Unif, 2 Orth-Gauss, 3 LeCun-Unif,
% 4 LeCun-Gauss, 5 Xavier, 6 MSRA/He.

if nargin < 3 || isempty(scal_factor)
    scal_factor = 1;
end
if numel(dim) < 2
    error('Please define at least input and output dimensions.');
end

dim = dim(:);
L = numel(dim) - 2;
W0(L+1).W = [];
b0(L+1).b = [];

switch init_type
    case 0
        for layer = 1:L+1
            W0(layer).W = zeros(dim(layer+1), dim(layer));
            b0(layer).b = zeros(dim(layer+1), 1);
        end
    case 1
        for layer = 1:L+1
            W0(layer).W = stiefel_init(-1 + 2 * rand(dim(layer+1), dim(layer)), scal_factor);
            b0(layer).b = zeros(dim(layer+1), 1);
        end
    case 2
        for layer = 1:L+1
            W0(layer).W = stiefel_init(randn(dim(layer+1), dim(layer)), scal_factor);
            b0(layer).b = zeros(dim(layer+1), 1);
        end
    case 3
        for layer = 1:L+1
            W0(layer).W = scal_factor * (-1 + 2 * rand(dim(layer+1), dim(layer))) * sqrt(3 / dim(layer));
            b0(layer).b = zeros(dim(layer+1), 1);
        end
    case 4
        for layer = 1:L+1
            W0(layer).W = scal_factor * randn(dim(layer+1), dim(layer)) * sqrt(1 / dim(layer));
            b0(layer).b = zeros(dim(layer+1), 1);
        end
    case 5
        for layer = 1:L+1
            W0(layer).W = scal_factor * (-1 + 2 * rand(dim(layer+1), dim(layer))) * ...
                sqrt(6 / (dim(layer+1) + dim(layer)));
            b0(layer).b = zeros(dim(layer+1), 1);
        end
    case 6
        for layer = 1:L
            W0(layer).W = scal_factor * randn(dim(layer+1), dim(layer)) * sqrt(2 / dim(layer));
            b0(layer).b = zeros(dim(layer+1), 1);
        end
        W0(L+1).W = scal_factor * randn(dim(L+2), dim(L+1)) * sqrt(1 / dim(L+1));
        b0(L+1).b = zeros(dim(L+2), 1);
    otherwise
        error('Unknown init_type.');
end
end

function W = stiefel_init(A, scal_factor)
if size(A, 1) < size(A, 2)
    Q = orth(A.');
    W = scal_factor * Q.';
else
    W = scal_factor * orth(A);
end
end
