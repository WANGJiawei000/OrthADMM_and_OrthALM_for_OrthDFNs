function V = V_Orth2(W2curr, U1curr, U2curr, b2curr, Lam12prev, Lam21prev, ...
                    beta, act_type, leak_slope) 
%V_ORTH2 Closed-form hidden-layer V update.


A = act_fun(U1curr, act_type, leak_slope) - Lam21prev / beta;

B = U2curr - b2curr - Lam12prev / beta;

C = W2curr.' * W2curr;

n = size(C, 1);

C(1:n+1:end) = C(1:n+1:end) + 1;

D = A + W2curr.' * B;

V = C \ D;

end
