function [approx_err, trainerr, cpu_time, iter_num, ypred, info, results] = Orth3_ALM_AM( ...
    x_train, x_test, y_train, train_y, test_y, W0, b0, act_type, niter, opts)
%ORTH3_ALM_AM Algorithm 7.5.1: ALM with alternating minimization for (7.3).
%
% This solver targets the regularized problem (7.3):
% data fitting + varpi1/2 * sum_l ||b_l||^2 + varpi2/2 * sum_l ||U_l||_F^2,
% subject to the network splitting constraints and Stiefel-copy constraints.
%
% Supported activations in the closed-form U update:
% act_type = 2 ReLU, 3 leaky ReLU, or 4 linear.


nLayers = numel(W0);

[~, N_train] = size(x_train);

if size(y_train, 2) ~= N_train

    error('x_train and y_train must have the same number of columns.');

end

varpi1 = get_option(opts, 'varpi1', 1e-6);
varpi2 = get_option(opts, 'varpi2', 1e-6);

mu1 = get_option(opts, 'mu1', 1e-5);
mu2 = get_option(opts, 'mu2', 1e-5);

beta = get_option(opts, 'beta0', 0.01);
betaMax = get_option(opts, 'betaMax', 1e8);

eta = get_option(opts, 'epsilon0', 1e-1);
etaMin = get_option(opts, 'epsilonMin', 1e-8);

eta1 = get_option(opts, 'eta1', 0.99);
eta2 = get_option(opts, 'eta2', 5/6);
eta3 = get_option(opts, 'eta3', 0.01);
eta4 = get_option(opts, 'eta4', 5/6);

tol = get_option(opts, 'tol', 1e-4);

minOuter = get_option(opts, 'minOuter', 3);

leak_slope = get_option(opts, 'leakSlope', 0.05);

verbose = get_option(opts, 'verbose', false);


if act_type == 2

    leak_slope = 0;

end


if varpi1 < 0 || varpi2 < 0

    error('varpi1 and varpi2 must be nonnegative.');

end



Z0(nLayers).Z = [];

Lam10(nLayers).Lam1 = [];
Lam20(nLayers).Lam2 = [];
Lam30(nLayers).Lam3 = [];

Cons1(nLayers).Cons1 = [];
Cons2(nLayers).Cons2 = [];
Cons3(nLayers).Cons3 = [];

Cons1N(nLayers).Cons1N = [];
Cons2N(nLayers).Cons2N = [];
Cons3N(nLayers).Cons3N = [];

stiefelCase = zeros(nLayers, 1);


for i = 1:nLayers

    stiefelCase(i) = get_stiefel_case(W0(i).W);

    if stiefelCase(i) == 1

        Lam30(i).Lam3 = zeros(size(W0(i).W));

        Cons3(i).Cons3 = zeros(size(W0(i).W));
        
        Z0(i).Z = W0(i).W;

    else
        
        Lam30(i).Lam3 = zeros(size(W0(i).W.'));

        Cons3(i).Cons3 = zeros(size(W0(i).W.'));
        
        Z0(i).Z = W0(i).W.';

    end

end


[U0, V0, ypred] = NN_fp(x_train, W0, b0, act_type, leak_slope);

for i = 1:nLayers

    Lam10(i).Lam1 = zeros(size(U0(i).U));

    Cons1(i).Cons1 = zeros(size(U0(i).U));

    Lam20(i).Lam2 = zeros(size(V0(i).V));

    Cons2(i).Cons2 = zeros(size(V0(i).V));

    Cons1N(i).Cons1N = 0;

    Cons2N(i).Cons2N = 0;

    Cons3N(i).Cons3N = 0;

end



feas1 = zeros(niter, 1);

feas2 = zeros(niter, 1);

feas3 = zeros(niter, 1);


trainerr = nan(niter, 1);

cpu_time = zeros(niter, 1);

acc_train = zeros(niter, 1);

acc_test = zeros(niter, 1);

regobj = nan(niter, 1);

constraintResidual = nan(niter, 1);

innerIterations = zeros(niter, 1);

betaHist = nan(niter, 1);

etaHist = nan(niter, 1);

LmaxHist = nan(niter, 1);

innerALStartHist = nan(niter, 1);

previousALHist = nan(niter, 1);

innerInitSource = cell(niter, 1);


trainerr(1) = norm(ypred - y_train, 'fro')^2 / N_train;

regobj(1) = regularized_objective(y_train, b0, U0, V0, varpi1, varpi2, N_train); % Objective function with regurized terms


[~, pred] = max(ypred);

correct_train = pred == train_y';

acc_train(1) = mean(correct_train);


[~, ~, ytestpred] = NN_fp(x_test, W0, b0, act_type, leak_slope);

[~, testpred] = max(ytestpred);

correct_test = testpred == test_y';

acc_test(1) = mean(correct_test);





Winit = W0;

Zinit = Z0;

binit = b0;

Uinit = U0;

Vinit = V0;

initialALValue = ALValue_ALM_Reg(regobj(1), Cons1, Cons2, Cons3, Cons1N, Cons2N, Cons3N, ...
        Lam10, Lam20, Lam30, beta);

cHat = initialALValue;




constraintResidual(1) = 0; % Constraint Residual \mathcal{R}^k

betaHist(1) = beta;

etaHist(1) = eta;

innerALStartHist(1) = initialALValue;

previousALHist(1) = initialALValue;

innerInitSource{1} = 'initial';

iter_num = 1;

previousResidual = constraintResidual(1);






for k = 2:niter


    %alm_start = tic;

    previousALValue = ALValue_ALM_Reg(regobj(k-1), Cons1, Cons2, Cons3, Cons1N, Cons2N, Cons3N, ...
        Lam10, Lam20, Lam30, beta);

    if k > 2 && previousALValue <= cHat

        Winner0 = W0;

        Zinner0 = Z0;

        binner0 = b0;

        Uinner0 = U0;

        Vinner0 = V0;

        initSource = 'previous';

        innerALValue = previousALValue;

    else

        Winner0 = Winit;

        Zinner0 = Zinit;

        binner0 = binit;

        Uinner0 = Uinit;

        Vinner0 = Vinit;

        initSource = 'original';

        innerALValue = initialALValue;

    end

    [Lmax, LmaxInfo] = Lmax_ALM_Reg(innerALValue, x_train, Winner0, ...
        Lam10, Lam20, Lam30, beta, varpi1, varpi2, mu1, mu2);


    innerOpts = opts;

    innerOpts.mu1 = mu1;
    innerOpts.mu2 = mu2;

    innerOpts.varpi1 = varpi1;
    innerOpts.varpi2 = varpi2;

    innerOpts.Lmax = Lmax;
    innerOpts.LmaxInfo = LmaxInfo;


    alm_start = tic;

    [W, Z, b, U, V, innerInfo] = AltMin_Orth3_Reg( ...
        x_train, y_train, Winner0, Zinner0, binner0, Uinner0, Vinner0, Lam10, Lam20, Lam30, ...
        beta, varpi1, varpi2, mu1, mu2, eta, innerOpts, ...
        stiefelCase, act_type, leak_slope); % BCD Algorithm 

    innerIterations(k) = innerInfo.iter;

    LmaxHist(k) = Lmax;

    innerALStartHist(k) = LmaxInfo.rhat;

    previousALHist(k) = previousALValue;

    innerInitSource{k} = initSource;

    eta = max(eta4 * eta, etaMin); % epsilon^k within ALM

    [Lam1, Lam2, Lam3, Cons1, Cons2, Cons3] = update_multipliers(x_train, W, Z, b, U, V, ...
        Lam10, Lam20, Lam30, beta, stiefelCase, act_type, leak_slope); % \Lambda update


    [constraintResidual(k), Cons1N, Cons2N, Cons3N] = compute_residual(Cons1, Cons2, Cons3);

    if constraintResidual(k) <= eta1 * previousResidual

        betaNext = beta;

    else

        betaNext = max(beta / eta2, max_multiplier_norm(Lam1, Lam2, Lam3)^(1 + eta3));

        betaNext = min(betaNext, betaMax);

    end

    W0 = W;
    Z0 = Z;
    b0 = b;
    U0 = U;
    V0 = V;

    Lam10 = Lam1;
    Lam20 = Lam2;
    Lam30 = Lam3;

    beta = betaNext;

    elapsed_time_alm = toc(alm_start);

    cpu_time(k) = cpu_time(k) + elapsed_time_alm;
    
    if k < niter

        cpu_time(k+1) = cpu_time(k);

    end

    [~, ~, ypred] = NN_fp(x_train, W, b, act_type, leak_slope);

    trainerr(k) = norm(ypred - y_train, 'fro')^2 / N_train;

    regobj(k) = regularized_objective(y_train, b, U, V, varpi1, varpi2, N_train);



    [~, pred] = max(ypred);

    correct_train = pred == train_y';

    acc_train(k) = mean(correct_train);
    
    
    [~, ~, ytestpred] = NN_fp(x_test, W, b, act_type, leak_slope);
    
    [~, testpred] = max(ytestpred);

    correct_test = testpred == test_y';

    acc_test(k) = mean(correct_test);



    betaHist(k) = beta;

    etaHist(k) = eta;

    iter_num = k;



    %% feasibility part

    feas1(k) = norm(W0(1).W * x_train + b0(1).b - U0(1).U, 'fro') / norm(U0(1).U, 'fro');

    for i = 2 : 1 : nLayers

        feas1(k) = feas1(k) + norm(W0(i).W * V0(i - 1).V + b0(i).b - U0(i).U, 'fro') / norm(U0(i).U, 'fro');

    end


    for i = 1 : 1 : nLayers

        feas2(k) = feas2(k) + norm(V0(i).V - act_fun(U0(i).U, act_type, leak_slope), 'fro') / norm(V0(i).V, 'fro');

        if stiefelCase(i) == 1

            stiefelResidual1 = W0(i).W - Z0(i).Z;

        else

            stiefelResidual1 = W0(i).W.' - Z0(i).Z;

        end

        feas3(k) = feas3(k) + norm(stiefelResidual1, 'fro') / norm(Z0(i).Z, 'fro');

    end


    %% end of feas part



    if verbose

        fprintf(['ALM outer %d, fit %.6e, reg obj %.6e, residual %.6e, ', ...
                 'beta %.3e, inner iters %d\n'], ...
            k, trainerr(k), regobj(k), constraintResidual(k), betaHist(k), innerIterations(k));

    end


    %if k >= minOuter && abs((regobj(k) - regobj(k-1))/ regobj(k-1)) <= tol

     %   break;

    %end

    previousResidual = constraintResidual(k);

end


approx_err = min(trainerr(1:iter_num));

info.W = W0;
info.b = b0;
info.Z = Z0;
info.U = U0;
info.V = V0;

info.Lam1 = Lam10;
info.Lam2 = Lam20;
info.Lam3 = Lam30;

info.trainerr = trainerr;

info.regobj = regobj;

info.constraintResidual = constraintResidual;

info.innerIterations = innerIterations;

info.betaHist = betaHist;

info.etaHist = etaHist;

info.LmaxHist = LmaxHist;

info.innerALStartHist = innerALStartHist;

info.previousALHist = previousALHist;

info.innerInitSource = innerInitSource(1:iter_num);

info.initialALValue = initialALValue;

info.cHat = cHat;

info.varpi1 = varpi1;
info.varpi2 = varpi2;

info.mu1 = mu1;
info.mu2 = mu2;

info.leakSlope = leak_slope;

info.stiefelCase = stiefelCase;

info.stiefelCopy = describe_stiefel_cases(stiefelCase);

info.algorithm = 'Algorithm 7.5.1 outer ALM with Algorithm 7.5.2 inner alternating minimization';



results.ApproxErr = approx_err;
results.TrainErr = trainerr;
results.TrainAcc = acc_train;
results.TestAcc = acc_test;
results.Time = cpu_time;
results.Iter = iter_num;
results.cons1 = feas1;
results.cons2 = feas2;
results.cons3 = feas3;


end







function value = get_option(opts, name, default_value)

if isfield(opts, name) && ~isempty(opts.(name))

    value = opts.(name);

else

    value = default_value;

end

end



function [Lam1, Lam2, Lam3, Cons1, Cons2, Cons3] = update_multipliers(X, W, Z, b, U, V, ...
    Lam10, Lam20, Lam30, beta, stiefelCase, act_type, leak_slope)

nLayers = numel(W);

Cons1(nLayers).Cons1 = [];
Cons2(nLayers).Cons2 = [];
Cons3(nLayers).Cons3 = [];

Lam1 = Lam10;
Lam2 = Lam20;
Lam3 = Lam30;

for i = 1:nLayers

    if i == 1

        Vprev = X;

    else

        Vprev = V(i-1).V;

    end

    Cons1(i).Cons1 = W(i).W * Vprev + b(i).b - U(i).U;

    Lam1(i).Lam1 = Lam10(i).Lam1 + beta * ...
        Cons1(i).Cons1;

    Cons2(i).Cons2 = V(i).V - act_fun(U(i).U, act_type, leak_slope);

    Lam2(i).Lam2 = Lam20(i).Lam2 + beta * ...
        Cons2(i).Cons2;

    if stiefelCase(i) == 1

        Cons3(i).Cons3 = W(i).W - Z(i).Z;

    else

        Cons3(i).Cons3 = W(i).W.' - Z(i).Z;

    end

    Lam3(i).Lam3 = Lam30(i).Lam3 + beta * Cons3(i).Cons3;

end

end



function [residual, Cons1N, Cons2N, Cons3N] = compute_residual(Cons1, Cons2, Cons3)

nLayers = numel(Cons1);

Cons1N(nLayers).Cons1N = [];
Cons2N(nLayers).Cons2N = [];
Cons3N(nLayers).Cons3N = [];

residual = 0;

for i = 1:nLayers

    Cons1N(i).Cons1N = norm(Cons1(i).Cons1, 'fro');

    residual = max(residual, Cons1N(i).Cons1N);

    Cons2N(i).Cons2N = norm(Cons2(i).Cons2, 'fro');

    residual = max(residual, Cons2N(i).Cons2N);

    Cons3N(i).Cons3N = norm(Cons3(i).Cons3, 'fro');

    residual = max(residual, Cons3N(i).Cons3N);

end

end




function obj = regularized_objective(Y, b, U, V, varpi1, varpi2, N_train)

nLayers = numel(U);

bterm = 0;

uterm = 0;

for i = 1:nLayers

    bterm = bterm + norm(b(i).b, 2)^2;

    uterm = uterm + norm(U(i).U, 'fro')^2;

end

obj = norm(V(nLayers).V - Y, 'fro')^2 / N_train + ...
    0.5 * varpi1 * bterm + 0.5 * varpi2 * uterm;

end



function maxNorm = max_multiplier_norm(Lam1, Lam2, Lam3)

maxNorm = 0;

for i = 1:numel(Lam1)

    maxNorm = max(maxNorm, norm(Lam1(i).Lam1, 'fro'));

    maxNorm = max(maxNorm, norm(Lam2(i).Lam2, 'fro'));
    
    maxNorm = max(maxNorm, norm(Lam3(i).Lam3, 'fro'));

end

end



function stiefel_case = get_stiefel_case(W)
if size(W, 1) >= size(W, 2)
    stiefel_case = 1;  % W_l = Z_l, so Z_l is tall or square.
else
    stiefel_case = 2;  % W_l.' = Z_l, so Z_l is tall or square.
end
end



function labels = describe_stiefel_cases(stiefelCase)
labels = cell(numel(stiefelCase), 1);
for i = 1:numel(stiefelCase)
    if stiefelCase(i) == 1
        labels{i} = 'W_l = Z_l';
    else
        labels{i} = 'W_l^T = Z_l';
    end
end
end
