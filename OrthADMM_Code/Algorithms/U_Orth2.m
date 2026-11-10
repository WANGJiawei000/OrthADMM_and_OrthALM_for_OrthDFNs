function U = U_Orth2(W1curr, V0prev, V1prev, b1curr, Lam11prev, Lam21prev, ...
                    beta, alpha, act_type) 
%U_ORTH2 Closed-form U update for piecewise affine activations.
% Supports ReLU, leaky ReLU, and linear activations.


A = W1curr * V0prev + b1curr + Lam11prev / beta;

B = V1prev + Lam21prev / beta;


switch act_type

    case 2

        neg_slope = 0;          % exact ReLU negative slope

    case 3

        neg_slope = alpha;      % leaky ReLU negative slope

    case 4

        U = (beta1 * A + beta2 * B) / (beta1 + beta2);

        return;

    otherwise

        error(['U_Orth2 has a closed-form update only for ReLU, leaky ReLU, ', ...
               'and linear activations. Sigmoid/sign need a different U-subproblem solver.']);

end

u_pos = max((A + B) / 2, 0);

if neg_slope == 0

    u_neg = min(A, 0);

else

    u_neg = min((A + neg_slope * B) / ...
                (1 + neg_slope^2), 0);

end

dpos_a = u_pos - A;

dpos_b = act_fun(u_pos, act_type, alpha) - B;

dneg_a = u_neg - A;

dneg_b = act_fun(u_neg, act_type, alpha) - B;

g_pos = (dpos_a .* dpos_a) + (dpos_b .* dpos_b);

g_neg = (dneg_a .* dneg_a) + (dneg_b .* dneg_b);

U = u_pos;

mask = g_neg < g_pos;

U(mask) = u_neg(mask);

end
