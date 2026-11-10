function U = U_ALM_Reg(Wcurr, Vprev, VcurrPrev, bcurr, Uprev, ...
    Lam1, Lam2, beta, varpi2, mu2, alpha, act_type)
%U_ALM_REG U_l update in Algorithm 7.5.2, equations (7.20)-(7.26).





C1 = Wcurr * Vprev + bcurr + Lam1 / beta;

C2 = VcurrPrev + Lam2 / beta;

switch act_type

    case 2

        neg_slope = 0;

    case 3

        neg_slope = alpha;

    case 4

        U = (beta * C1 + beta * C2 + mu2 * Uprev) / (2 * beta + mu2 + varpi2);

        return;

    otherwise

        error(['U_ALM_Reg has a closed-form update only for ReLU, leaky ReLU, ', ...
               'and linear activations. Sigmoid/sign need a different scalar solver.']);

end


u_pos = max((beta * (C1 + C2) + mu2 * Uprev) / ...
    (2 * beta + mu2 + varpi2), 0);

if neg_slope == 0

    u_neg = min((beta * C1 + mu2 * Uprev) / ...
        (beta + mu2 + varpi2), 0);

else

    u_neg = min((beta * C1 + neg_slope * beta * C2 + mu2 * Uprev) / ...
        (beta + neg_slope^2 * beta + mu2 + varpi2), 0);

end

g_pos = varpi2 .* (u_pos .* u_pos) + ...
    beta .* ((u_pos - C1) .* (u_pos - C1)) + ...
    beta .* ((act_fun(u_pos, act_type, alpha) - C2) .* ...
              (act_fun(u_pos, act_type, alpha) - C2)) + ...
    mu2 .* ((u_pos - Uprev) .* (u_pos - Uprev));

g_neg = varpi2 .* (u_neg .* u_neg) + ...
    beta .* ((u_neg - C1) .* (u_neg - C1)) + ...
    beta .* ((act_fun(u_neg, act_type, alpha) - C2) .* ...
              (act_fun(u_neg, act_type, alpha) - C2)) + ...
    mu2 .* ((u_neg - Uprev) .* (u_neg - Uprev));

U = u_pos;

mask = g_neg < g_pos;

U(mask) = u_neg(mask);

end
