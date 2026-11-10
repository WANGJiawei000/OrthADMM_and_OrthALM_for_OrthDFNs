function V = Vend_Orth2(Y, U1curr, Lam21prev, beta, N_train, act_type, leak_slope)
%VEND_ORTH2 Closed-form output-layer V update for problem (7.2).

if nargin < 7 || isempty(leak_slope)

    leak_slope = 0.05;

end

E = beta * act_fun(U1curr, act_type, leak_slope) - Lam21prev;

V = (2 * Y + N_train * E) / (2 + N_train * beta);

end
