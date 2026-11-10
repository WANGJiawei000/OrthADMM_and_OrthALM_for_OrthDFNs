function [approx_err, trainerr, cpu_time, iter_num, ypred, info, results] = Orth3_ADMM( ...
    x_train, x_test, y_train, train_y, test_y, W0, b0, act_type, niter, opts)
%ORTH3_ADMM ADMM solver for the unregularized OrthDNN problem (7.2).
%
% act_type: 2 ReLU, 3 leaky ReLU, or 4 linear are supported by the closed-form
% U update in U_Orth2. Sigmoid/sign require a different U-subproblem solver.


if ~ismember(act_type, [2, 3, 4])

    error(['This ADMM implementation supports act_type 2 (ReLU), 3 (leaky ReLU), ', ...
           'and 4 (linear). Sigmoid/sign need a different U update.']);

end

L = numel(W0) - 1;

[~, N_train] = size(x_train);


beta = get_option(opts, 'beta', 0.05);

innerSweeps = get_option(opts, 'innerSweeps', 1);

tol = get_option(opts, 'tol', 1e-4);

leak_slope = get_option(opts, 'leakSlope', 0.05);

verbose = get_option(opts, 'verbose', false);


if act_type == 2

    leak_slope = 0;

end


trainerr = zeros(niter, 1);

acc_train = zeros(niter, 1);

acc_test = zeros(niter, 1);

cpu_time = zeros(niter, 1);

feas1 = zeros(niter, 1);

feas2 = zeros(niter, 1);

feas3 = zeros(niter, 1);


Z0(L+1).Z = [];

Lam10(L+1).Lam1 = [];
Lam20(L+1).Lam2 = [];
Lam30(L+1).Lam3 = [];

stiefelCase = zeros(L+1, 1);

for i = 1:L+1

    stiefelCase(i) = get_stiefel_case(W0(i).W);

    if stiefelCase(i) == 1

        Lam30(i).Lam3 = zeros(size(W0(i).W));

        Z0(i).Z = W0(i).W;

    else

        Lam30(i).Lam3 = zeros(size(W0(i).W.'));

        Z0(i).Z = W0(i).W.';

    end

end


[U0, V0, ypred_train] = NN_fp(x_train, W0, b0, act_type, leak_slope);

for i = 1:L+1

    Lam10(i).Lam1 = zeros(size(U0(i).U));

    Lam20(i).Lam2 = zeros(size(V0(i).V));

end

trainerr(1) = norm(ypred_train - y_train, 'fro')^2 / N_train;

iter_num = 1;


[~, pred] = max(ypred_train);

correct_train = pred == train_y';

acc_train(1) = mean(correct_train);


[~, ~, ytestpred] = NN_fp(x_test, W0, b0, act_type, leak_slope);

[~, testpred] = max(ytestpred);

correct_test = testpred == test_y';

acc_test(1) = mean(correct_test);





for k = 2 : niter

    admm_start = tic;

    for sweep = 1 : innerSweeps 

        W = W0;
        Z = Z0;
        b = b0;
        U = U0;
        V = V0;

        for i = L+1:-1:2

            W(i).W = W_Orth3(Z0(i).Z, b0(i).b, V0(i-1).V, U0(i).U, ...
                Lam10(i).Lam1, Lam30(i).Lam3, beta, ...
                stiefelCase(i));

        end

        W(1).W = W_Orth3(Z0(1).Z, b0(1).b, x_train, U0(1).U, ...
            Lam10(1).Lam1, Lam30(1).Lam3, beta, ...
            stiefelCase(1));

        for i = L+1:-1:1

            Z(i).Z = Z_Orth3(W(i).W, beta, Lam30(i).Lam3, stiefelCase(i));

        end

        for i = L+1:-1:2

            b(i).b = b_Orth2(W(i).W, V0(i-1).V, U0(i).U, ...
                Lam10(i).Lam1, beta, N_train);

        end

        b(1).b = b_Orth2(W(1).W, x_train, U0(1).U, ...
            Lam10(1).Lam1, beta, N_train);

        for i = 1:L

            V(i).V = V_Orth2(W(i+1).W, U0(i).U, U0(i+1).U, b(i+1).b, ...
                Lam10(i+1).Lam1, Lam20(i).Lam2, beta, ...
                act_type, leak_slope);

        end

        V(L+1).V = Vend_Orth2(y_train, U0(L+1).U, Lam20(L+1).Lam2, ...
            beta, N_train, act_type, leak_slope);


        U(1).U = U_Orth2(W(1).W, x_train, V(1).V, b(1).b, ...
            Lam10(1).Lam1, Lam20(1).Lam2, beta, ...
            leak_slope, act_type);

        for i = 2:L+1

            U(i).U = U_Orth2(W(i).W, V(i-1).V, V(i).V, b(i).b, ...
                Lam10(i).Lam1, Lam20(i).Lam2, beta, ...
                leak_slope, act_type);

        end


        W0 = W;
        Z0 = Z;
        b0 = b;
        U0 = U;
        V0 = V;

    end


    Lam1 = Lam10;
    Lam2 = Lam20;
    Lam3 = Lam30;

    Lam1(1).Lam1 = Lam10(1).Lam1 + beta * ...
        (W(1).W * x_train + b(1).b - U(1).U);

    for i = 2:L+1

        Lam1(i).Lam1 = Lam10(i).Lam1 + beta * ...
            (W(i).W * V(i-1).V + b(i).b - U(i).U);

    end

    for i = 1:L+1

        Lam2(i).Lam2 = Lam20(i).Lam2 + beta * ...
            (V(i).V - act_fun(U(i).U, act_type, leak_slope));

        if stiefelCase(i) == 1

            stiefelResidual = W(i).W - Z(i).Z;

        else

            stiefelResidual = W(i).W.' - Z(i).Z;

        end

        Lam3(i).Lam3 = Lam30(i).Lam3 + beta * stiefelResidual;

    end

    Lam10 = Lam1;
    Lam20 = Lam2;
    Lam30 = Lam3;


    elapsed_time_admm = toc(admm_start);

    cpu_time(k) = cpu_time(k) + elapsed_time_admm;
    
    if k < niter

        cpu_time(k+1) = cpu_time(k);

    end



    [~, ~, ypred] = NN_fp(x_train, W0, b0, act_type, leak_slope);
    
    trainerr(k) = norm(ypred - y_train, 'fro')^2 / N_train;

    iter_num = k;


    [~, pred] = max(ypred);

    correct_train = pred == train_y';

    acc_train(k) = mean(correct_train);
    
    
    [~, ~, ytestpred] = NN_fp(x_test, W0, b0, act_type, leak_slope);
    
    [~, testpred] = max(ytestpred);

    correct_test = testpred == test_y';

    acc_test(k) = mean(correct_test);


    %% feasibility part

    feas1(k) = norm(W0(1).W * x_train + b0(1).b - U0(1).U, 'fro') / norm(U0(1).U, 'fro');

    for i = 2 : 1 : L+1

        feas1(k) = feas1(k) + norm(W0(i).W * V0(i - 1).V + b0(i).b - U0(i).U, 'fro') / norm(U0(i).U, 'fro');

    end


    for i = 1 : 1 : L+1

        feas2(k) = feas2(k) + norm(V0(i).V - act_fun(U0(i).U, act_type, leak_slope), 'fro') / norm(V0(i).V, 'fro');

        if stiefelCase(i) == 1

            stiefelResidual1 = W0(i).W - Z0(i).Z;

        else

            stiefelResidual1 = W0(i).W.' - Z0(i).Z;

        end

        feas3(k) = feas3(k) + norm(stiefelResidual1, 'fro') / norm(Z0(i).Z, 'fro');

    end


    %% end of feas part

    if verbose && (mod(k, 10) == 0 || k == 2)

        fprintf('ADMM iter %d, train error %.6e\n', k, trainerr(k));

    end


    if k > 3

        denom = max(abs(trainerr(k-1)), eps);

        if abs(trainerr(k) - trainerr(k-1)) / denom <= tol

            break;

        end

    end

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
info.beta = beta;
info.innerSweeps = innerSweeps;
info.leakSlope = leak_slope;
info.stiefelCase = stiefelCase;
info.stiefelCopy = describe_stiefel_cases(stiefelCase);


results.ApproxErr = approx_err;
results.TrainErr = trainerr;
results.TrainAcc = acc_train;
results.TestAcc = acc_test;
results.Time = cpu_time;
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
