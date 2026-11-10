function b = b_Orth2(W1curr, V0prev, U1prev, Lam11prev, beta, N_train)
%B_ORTH2 Bias update for problem (7.2).
% Uses column sums to avoid forming W1curr*V0prev as a full n_l by N matrix.

if nargin < 6 || isempty(N_train)
    N_train = size(V0prev, 2);
end

sumV = sum(V0prev, 2);

b = (sum(U1prev, 2) - W1curr * sumV - sum(Lam11prev, 2) / beta) / N_train;

end
