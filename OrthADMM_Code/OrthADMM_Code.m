clearvars;
close all;
clc;


if exist('Algorithms', 'dir')

    addpath('Algorithms');

end

if exist('Tools', 'dir')

    addpath('Tools');

end


%%

[x_train, y_train, x_test, y_test, train_y, test_y] = mnist_fixed_small_format( ...
    '/Users/jiaweiwang/mnist_common_small_data/train_x_small_500.mat', ...
    '/Users/jiaweiwang/mnist_common_small_data/train_y_small_500.mat', ...
    '/Users/jiaweiwang/mnist_common_small_data/test_x_small_200.mat', ...
    '/Users/jiaweiwang/mnist_common_small_data/test_y_small_200.mat');


%%

L = 2;
dim = zeros(L+2, 1);
dim(1) = size(x_train, 1);
dim(2) = 50;
dim(3) = 25;
dim(L+2) = size(y_train, 1);

[W0, b0] = init_scheme(dim, 2, 1);


acttype = 2;              % 2 ReLU, 3 leaky ReLU, 4 linear
%acttype = 3; 

NumEpoch = 200;


opts.beta = 0.002;  

opts.innerSweeps = 1;     % set to 5 to reproduce your old inner loop

%opts.tol = 1e-4;
opts.tol = 1e-7;

opts.leakSlope = 0.01;    % used only when acttype = 3
%opts.leakSlope = 0.1; 

opts.verbose = true;

fprintf('Starting ADMM for OrthDNNs, acttype = %d\n', acttype);


[ApproxErr, trainerr, cpu_time, iter_num, ypred, info, results] = Orth3_ADMM( ...
    x_train, x_test, y_train, train_y, test_y, W0, b0, acttype, NumEpoch, opts);

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
