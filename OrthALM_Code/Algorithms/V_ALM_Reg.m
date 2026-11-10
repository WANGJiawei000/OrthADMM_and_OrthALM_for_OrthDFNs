function V = V_ALM_Reg(Wnext, Ucurr, Unext, bnext, Lam1next, Lam2, ...
    beta, act_type, leak_slope)
%V_ALM_REG hidden-layer V_l update in Algorithm 7.5.2, equation (7.28).



A = act_fun(Ucurr, act_type, leak_slope) - Lam2 / beta;

B = Unext - bnext - Lam1next / beta;

C = Wnext.' * Wnext;

n = size(C, 1);

C(1:n+1:end) = C(1:n+1:end) + 1;

D = A + Wnext.' * B;

V = C \ D;

end
