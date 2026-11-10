function Z = Z_Orth3(W1curr, beta, Lam31prev, stiefel_case)
%Z_ORTH3 Projection onto the rectangular Stiefel set.
%
% stiefel_case = 1: project W_l   + Lambda_3/beta_3, with W_l tall or square.
% stiefel_case = 2: project W_l.' + Lambda_3/beta_3, with W_l fat.

if nargin < 4 || isempty(stiefel_case)

    stiefel_case = 1;

end

if stiefel_case == 1

    F = W1curr + Lam31prev / beta;

elseif stiefel_case == 2

    F = W1curr.' + Lam31prev / beta;

else

    error('stiefel_case must be 1 (W_l = Z_l) or 2 (W_l.'' = Z_l).');

end

[U, ~, V] = svd(F, 'econ');

Z = U * V.';

end
