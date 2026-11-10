function value = ALValue_ALM_Reg(regobj, Cons1, Cons2, Cons3, Cons1N, Cons2N, Cons3N, ...
    Lam1, Lam2, Lam3, beta)
% ALVALUE_ALM_REG Augmented Lagrangian value for problem (7.3).
%
% The residual signs are chosen to match the update formulas in this folder:
%   W_l V_{l-1} + b_l 1_m^T - U_l,
%   V_l - sigma(U_l),
%   W_l - Z_l or W_l.' - Z_l.

nLayers = numel(Cons1);

dual = 0;

penalty = 0;


for l = 1:nLayers

    dual = dual + sum(sum(Lam1(l).Lam1 .* Cons1(l).Cons1)) + ...
        sum(sum(Lam2(l).Lam2 .* Cons2(l).Cons2)) + sum(sum(Lam3(l).Lam3 .* Cons3(l).Cons3));

    penalty = penalty + (Cons1N(l).Cons1N)^2 + (Cons2N(l).Cons2N)^2 + (Cons3N(l).Cons3N)^2;

end


parts.regobj = regobj;

parts.dual = dual;

parts.penalty = 0.5 * beta * penalty;

value = parts.regobj + parts.dual + parts.penalty;


end
