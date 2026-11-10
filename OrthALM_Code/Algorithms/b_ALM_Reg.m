function b = b_ALM_Reg(Wcurr, Vprev, Uprev, Lam1, beta, varpi1, N_train)
%B_ALM_REG b_l update in Algorithm 7.5.2, equation (7.19).



sumV = sum(Vprev, 2);

numerator = beta * (sum(Uprev, 2) - Wcurr * sumV) - sum(Lam1, 2);

b = numerator / (varpi1 + N_train * beta);

end
