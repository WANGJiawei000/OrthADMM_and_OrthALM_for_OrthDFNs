clearvars;
close all;
clc;

if exist('Algorithms', 'dir')
    addpath('Algorithms');
end

if exist('Tools', 'dir')
    addpath('Tools');
end


[x_train, y_train, x_test, y_test, train_y, test_y] = mnist_fixed_small_format( ...
    '/Users/jiaweiwang/mnist_common_small_data/train_x_small_500.mat', ...
    '/Users/jiaweiwang/mnist_common_small_data/train_y_small_500.mat', ...
    '/Users/jiaweiwang/mnist_common_small_data/test_x_small_200.mat', ...
    '/Users/jiaweiwang/mnist_common_small_data/test_y_small_200.mat');



L = 2;
dim = zeros(L+2, 1);
dim(1) = size(x_train, 1);
dim(2) = 50;
dim(3) = 25;
dim(L+2) = size(y_train, 1);

[W0, b0] = init_scheme(dim, 2, 1);

acttype = 2;              % 2 ReLU, 3 leaky ReLU, 4 linear 121s 0.9441
%acttype = 3; 

MaxOuter = 200;          % outer ALM iterations

opts.varpi1 = 1e-8;      % regularization parameter for sum ||b_l||^2
%opts.varpi1 = 1e-7;
opts.varpi2 = 1e-8;      % regularization parameter for sum ||U_l||_F^2
%opts.varpi2 = 1e-7;

opts.mu1 = 1e-4;         % proximal parameter for Z update in Algorithm 7.5.2
%opts.mu1 = 1e-3;
opts.mu2 = 1e-4;         % proximal parameter for U update in Algorithm 7.5.2
%opts.mu2 = 1e-3;

opts.beta0 = 0.006;       % initial ALM penalty beta^0
%opts.beta0 = 0.001;
opts.betaMax = 1e8;

opts.epsilon0 = 1e-1;    % initial inner AM tolerance epsilon_0
opts.epsilonMin = 1e-8;

%opts.eta1 = 0.99;         % residual shrinkage test in Algorithm 7.5.1
opts.eta1 = 0.99;
opts.eta2 = 0.99;         % beta update uses beta/eta2 when residual stalls
%opts.eta2 = 0.8;
%opts.eta3 = 0.01;         % multiplier exponent parameter, eta3 > 1
%opts.eta3 = 0.5;
opts.eta3 = 1.5;
opts.eta4 = 0.9;         % inner tolerance reduction epsilon_k = eta4 epsilon_{k-1}
%opts.eta4 = 0.5;

opts.minInner = 3;       % min inner alternating-minimization iterations
opts.maxInner = 7;      % max inner alternating-minimization iterations
opts.minOuter = 3;       % min outer ALM iterations

%opts.tol = 1e-4;         % stopping tolerance for the ALM constraint residual
opts.tol = 1e-5;

opts.leakSlope = 0.01;    % used only when acttype = 3
%opts.leakSlope = 0.05;


opts.verbose = true;

opts.innerVerbose = false;


fprintf('Starting ALM-AM for regularized OrthDNNs, acttype = %d\n', acttype);



[ApproxErr, trainerr, cpu_time, iter_num, ypred, info, results] = Orth3_ALM_AM( ...
    x_train, x_test, y_train, train_y, test_y, W0, b0, acttype, MaxOuter, opts);

time_total = cpu_time(iter_num);



[~, pred] = max(ypred);
correct_train = pred == train_y';
acc_train = mean(correct_train);


[~, ~, ytestpred] = NN_fp(x_test, info.W, info.b, acttype, opts.leakSlope);

[~, testpred] = max(ytestpred);
correct_test = testpred == test_y';
acc_test = mean(correct_test);


fprintf('approximation error: %g, running time: %g\n', ApproxErr, time_total);

fprintf('training accuracy: %.4f, test accuracy: %.4f\n', acc_train, acc_test);

figure;
Epoch = (1:iter_num)';
semilogy(Epoch, trainerr(1:iter_num), 'r-', 'LineWidth', 2);
xlabel('Epoch number', 'FontSize', 18);
ylabel('Approximation error', 'FontSize', 18);
grid on;
