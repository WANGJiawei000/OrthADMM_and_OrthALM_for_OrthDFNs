function [W, Z, b, U, V, info] = AltMin_Orth3_Reg( ...
    X, Y, W0, Z0, b0, U0, V0, Lam1, Lam2, Lam3, beta, ...
    varpi1, varpi2, mu1, mu2, epsilon, opts, stiefelCase, act_type, leak_slope)
%ALTMIN_ORTH3_REG Algorithm 7.5.2 for the ALM subproblem (7.7).
%
% This inner solver alternates W, Z, b, V, and U updates for fixed
% multiplier matrices Lambda and fixed ALM penalty beta.


nLayers = numel(W0);

[~, N_train] = size(X);

maxInner = get_option(opts, 'maxInner', 10);

minInner = get_option(opts, 'minInner', 3);

innerVerbose = get_option(opts, 'innerVerbose', false);

Lmax = opts.Lmax;


W = W0;
Z = Z0;
b = b0;
U = U0;
V = V0;

changeHist = nan(maxInner, 1);

threshold = epsilon^2 / max(Lmax, eps);


for j = 1:maxInner

    %Wprev = W;
    Zprev = Z;
    bprev = b;
    Uprev = U;
    Vprev = V;

    for i = 1:nLayers

        if i == 1

            Vinput = X;

        else

            Vinput = Vprev(i-1).V;

        end

        W(i).W = W_ALM_Reg(Zprev(i).Z, bprev(i).b, Vinput, Uprev(i).U, ...
            Lam1(i).Lam1, Lam3(i).Lam3, beta, stiefelCase(i));

    end


    for i = 1:nLayers

        Z(i).Z = Z_ALM_Reg(W(i).W, Zprev(i).Z, Lam3(i).Lam3, ...
            beta, mu1, stiefelCase(i));

    end


    for i = 1:nLayers

        if i == 1

            Vinput = X;

        else

            Vinput = Vprev(i-1).V;

        end

        b(i).b = b_ALM_Reg(W(i).W, Vinput, Uprev(i).U, ...
            Lam1(i).Lam1, beta, varpi1, N_train);

    end


    for i = 1:nLayers-1

        V(i).V = V_ALM_Reg(W(i+1).W, Uprev(i).U, Uprev(i+1).U, b(i+1).b, ...
            Lam1(i+1).Lam1, Lam2(i).Lam2, beta, act_type, leak_slope);

    end

    V(nLayers).V = Vend_ALM_Reg(Y, Uprev(nLayers).U, Lam2(nLayers).Lam2, ...
        beta, N_train, act_type, leak_slope);


    for i = 1:nLayers

        if i == 1

            Vinput = X;

        else

            Vinput = V(i-1).V;

        end

        U(i).U = U_ALM_Reg(W(i).W, Vinput, V(i).V, b(i).b, Uprev(i).U, ...
            Lam1(i).Lam1, Lam2(i).Lam2, beta, varpi2, mu2, leak_slope, act_type);

    end

    change = inner_change(Z, Zprev, b, bprev, U, Uprev, V, Vprev);

    changeHist(j) = change;

    if innerVerbose

        fprintf('  AM inner %d, change %.6e, threshold %.6e\n', j, change, threshold);

    end

    if j >= minInner && change <= threshold

        break;
    
    end

end

info.iter = j;

info.change = changeHist(1:j);

info.lastChange = changeHist(j);

info.threshold = threshold;

info.Lmax = Lmax;

end



function value = get_option(opts, name, default_value)

if isfield(opts, name) && ~isempty(opts.(name))

    value = opts.(name);

else

    value = default_value;

end

end



function change = inner_change(Z, Zprev, b, bprev, U, Uprev, V, Vprev)

change = 0;

for i = 1:numel(Z)

    change = change + norm(Z(i).Z - Zprev(i).Z, 'fro')^2;

    change = change + norm(b(i).b - bprev(i).b, 2)^2;

    change = change + norm(U(i).U - Uprev(i).U, 'fro')^2;

    change = change + norm(V(i).V - Vprev(i).V, 'fro')^2;

end

end
