function Z = Z_ALM_Reg(Wcurr, Zprev, Lam3, beta, mu1, stiefel_case)
%Z_ALM_REG Z_l update in Algorithm 7.5.2, equations (7.16)-(7.17).


if stiefel_case == 1

    C = Wcurr;

elseif stiefel_case == 2

    C = Wcurr.';

else

    error('stiefel_case must be 1 (W_l = Z_l) or 2 (W_l.'' = Z_l).');

end

F = (beta * C + Lam3 + mu1 * Zprev) / (beta + mu1);

[U, ~, V] = svd(F, 'econ');

Z = U * V.';

end
