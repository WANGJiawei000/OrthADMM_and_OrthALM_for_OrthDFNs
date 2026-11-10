function W = W_Orth3(Z1prev, b1prev, V0prev, U1prev, Lam11prev, Lam31prev, ...
                    beta, stiefel_case) 
%W_ORTH3 Closed-form W update for either Stiefel-copy constraint.
%
% stiefel_case = 1: W_l   = Z_l, used when W_l is tall or square.
% stiefel_case = 2: W_l.' = Z_l, used when W_l is fat so Z_l is tall.

A  = U1prev - b1prev - Lam11prev / beta;

AV = A * V0prev.';

VV = V0prev * V0prev.';

if stiefel_case == 1

    B = Z1prev - Lam31prev / beta;

elseif stiefel_case == 2

    B = (Z1prev - Lam31prev / beta).';

else

    error('stiefel_case must be 1 (W_l = Z_l) or 2 (W_l.'' = Z_l).');

end

n = size(VV, 1);

VV(1:n+1:end) = VV(1:n+1:end) + 1;

W = (AV + B) / VV;

end
