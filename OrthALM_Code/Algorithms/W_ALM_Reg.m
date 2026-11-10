function W = W_ALM_Reg(Zprev, bprev, Vprev, Uprev, Lam1, Lam3, beta, stiefel_case)
%W_ALM_REG W_l update in Algorithm 7.5.2, equation (7.15).


A = Uprev - bprev - Lam1 / beta;

AV = A * Vprev.';

VV = Vprev * Vprev.';

if stiefel_case == 1

    B = Zprev - Lam3 / beta;

elseif stiefel_case == 2

    B = (Zprev - Lam3 / beta).';

else

    error('stiefel_case must be 1 (W_l = Z_l) or 2 (W_l.'' = Z_l).');

end

n = size(VV, 1);

VV(1:n+1:end) = VV(1:n+1:end) + 1;

W = (AV + B) / VV;

end
