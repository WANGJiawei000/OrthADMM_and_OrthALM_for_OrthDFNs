function V = Vend_ALM_Reg(Y, Ucurr, Lam2, beta, N_train, act_type, leak_slope)
%VEND_ALM_REG output-layer V_l update in Algorithm 7.5.2, equation (7.29).



E = beta * act_fun(Ucurr, act_type, leak_slope) - Lam2;

V = (2 * Y + N_train * E) / (2 + N_train * beta);

end
