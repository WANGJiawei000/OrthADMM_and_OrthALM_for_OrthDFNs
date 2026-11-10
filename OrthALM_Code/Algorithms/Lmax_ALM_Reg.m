function [Lmax, info] = Lmax_ALM_Reg(innerALValue, X, W, ...
    Lam1, Lam2, Lam3, beta, varpi1, varpi2, mu1, mu2)
%LMAX_ALM_REG Bounding parameter L_max^{k-1} from equation (7.13).
%
% This implements equation (7.13) using the constants L_{1,l}, L_{2,l},
% and L_{3,l} defined in Lemma 8.1.1, equation (8.4).


if beta <= 0

    error('beta must be positive when computing Lmax.');

end

if varpi1 <= 0 || varpi2 <= 0

    error('varpi1 and varpi2 must be positive for the Lmax formula in Lemma 8.1.1.');

end

if mu1 <= 0 || mu2 <= 0

    error('mu1 and mu2 must be positive for the Lmax formula in Lemma 8.1.1.');

end


nLayers = numel(W);

m = size(X, 2);

nmax = size(X, 1);

lambdaSqSum = 0;

Lam1N(nLayers).Lam1N = [];

Lam2N(nLayers).Lam2N = [];

Lam3N(nLayers).Lam3N = [];


for l = 1:nLayers

    nmax = max(nmax, size(W(l).W, 1));

    Lam1N(l).Lam1N = norm(Lam1(l).Lam1, 'fro');

    Lam2N(l).Lam2N = norm(Lam2(l).Lam2, 'fro');

    Lam3N(l).Lam3N = norm(Lam3(l).Lam3, 'fro');

    lambdaSqSum = lambdaSqSum + Lam1N(l).Lam1N^2 + ...
        Lam2N(l).Lam2N^2 + Lam3N(l).Lam3N^2;

end

rhat = innerALValue;

delta = rhat + lambdaSqSum / (2 * beta);

delta = max(delta, 0);

delta0 = zeros(nLayers, 1);

delta1 = zeros(nLayers, 1);

delta2 = zeros(nLayers, 1);

L1 = zeros(nLayers, 1);

L2 = zeros(nLayers, 1);

L3 = zeros(nLayers, 1);


sqrt2deltabeta = sqrt(2 * delta / beta); 

delta0cons = sqrt2deltabeta + sqrt(nmax);

sqrt2deltavarpi2 = sqrt(2 * delta / varpi2);

delta1cons = sqrt2deltabeta + sqrt2deltavarpi2;

for l = 1:nLayers

    delta0(l) = (delta0cons + Lam3N(l).Lam3N / beta)^2;

    delta1(l) = (delta1cons + Lam2N(l).Lam2N / beta)^2;

end

beta2 = beta^2;

for l = 1:nLayers

    if l == 1

        L1(l) = 3 * beta2 * max(m * norm(X, 'fro')^2, 1);

        L2(l) = m * beta2;

    else

        delta2(l) = Lam1N(l).Lam1N^2 + beta2 * ...
            (2 * delta0(l) * delta1(l-1) + 2 * m * delta / varpi1 + ...
             2 * delta / varpi2);

        L1(l) = 8 * max([delta2(l), beta2, m * beta2 * delta1(l-1)]);

        L2(l) = 2 * m * beta2 * max(delta0(l), 1);

    end

    if l < nLayers

        L3(l) = 2 * beta2 * max(delta0(l+1), 1);

    else

        L3(l) = beta2;

    end

end


firstBlock = max(L1) + mu1^2;

if nLayers > 1

    secondBlock = max(L1(2:end) + L2(2:end));

else

    secondBlock = 0;

end

thirdBlock = max(L1 + L2 + L3 + [0; L3(1:end-1)]) + mu2^2;


Lmax = max([firstBlock, secondBlock, thirdBlock]);


info.rhat = rhat;

info.delta = delta;

info.delta0 = delta0;

info.delta1 = delta1;

info.delta2 = delta2;

info.L1 = L1;

info.L2 = L2;

info.L3 = L3;

info.firstBlock = firstBlock;

info.secondBlock = secondBlock;

info.thirdBlock = thirdBlock;

info.nmax = nmax;


end
